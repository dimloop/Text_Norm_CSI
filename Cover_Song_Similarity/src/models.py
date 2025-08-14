import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class Conv1dBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

class ResBlock1D(nn.Module):
    """
    Two Conv-BN-ReLU (+dropout) with a residual/skip connection.
    Keeps temporal length the same (stride=1, padding=1).
    Uses a 1x1 projection if channels change.
    """
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv1 = Conv1dBNReLU(in_ch, out_ch, k=3, s=1, p=1, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.proj = (nn.Identity() if in_ch == out_ch
                     else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.proj(identity)
        return out

class FeatureNet(nn.Module):
    """
    Input: [B, T, F]  (e.g., F=24 for HPCP+CREMA or F=12 for HPCP only)
    Output: L2-normalized embedding [B, embedding_size]
    """
    def __init__(self, input_dims=24, embedding_size=128, dropout=0.2):
        super().__init__()

        self.stem = nn.Identity()  # we keep your transpose pattern in forward

        # Replace your ConvBlock with ResBlock1D, keep pooling after each block
        self.block1 = ResBlock1D(input_dims, 32,  dropout=dropout)
        self.pool1  = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block2 = ResBlock1D(32,  64,  dropout=dropout)
        self.pool2  = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block3 = ResBlock1D(64,  128, dropout=dropout)
        self.pool3  = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block4 = ResBlock1D(128, 256, dropout=dropout)
        self.pool4  = nn.MaxPool1d(kernel_size=4, stride=4)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):           # x: [B, T, F]
        x = x.transpose(1, 2)       # -> [B, F, T] for Conv1d

        x = self.block1(x); x = self.pool1(x)
        x = self.block2(x); x = self.pool2(x)
        x = self.block3(x); x = self.pool3(x)
        x = self.block4(x); x = self.pool4(x)

        avg = self.global_avg_pool(x).squeeze(-1)  # [B,256]
        mx  = self.global_max_pool(x).squeeze(-1)  # [B,256]
        z   = torch.cat([avg, mx], dim=1)          # [B,512]

        emb = self.fc_layers(z)                    # [B,embedding_size]
        #emb = F.normalize(emb, p=2, dim=1)
        return emb


##################################################################################

class QuantumBlock(nn.Module):
    """
    AmplitudeEmbedding (length = 2^n_qubits) + BasicEntanglerLayers -> expvals (n_qubits)
    Then project back to embedding_size so downstream code stays the same.
    """
    def __init__(self, embedding_size=128, n_qubits=7, n_layers=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_qubits = n_qubits
        self.amp_size = 2 ** n_qubits  # 128 when n_qubits=7

        # Build QNode
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def qnode(inputs, weights):
            # inputs: [amp_size], dtype float64
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (n_layers, self.n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)  # returns float64 [B, n_qubits]

        # map expvals [B, n_qubits] back to [B, embedding_size]
        self.post = nn.Sequential(
            nn.Linear(self.n_qubits, embedding_size, dtype=torch.float64),
            nn.ReLU(),
        )

    def forward(self, emb):  # emb: [B, embedding_size] (float32)
        # If embedding_size != amp_size, learn a projection to amp_size
        if emb.shape[1] != self.amp_size:
            proj = nn.Linear(emb.shape[1], self.amp_size, bias=True).to(emb.device).to(emb.dtype)
            # NOTE: for JIT/static graph you'd register this layer in __init__ if sizes are known.
            emb_amp = proj(emb)
        else:
            emb_amp = emb
        # Cast to float64 for PennyLane
        qin   = emb_amp.to(torch.float64)
        q_out = self.qlayer(qin)               # [B, n_qubits], float64
        z     = self.post(q_out)               # [B, embedding_size], float64
        z     = z.to(torch.float32)
        return z#F.normalize(z, p=2, dim=1)      # keep unit-norm embeddings

# -------------------------
# Siamese with quantum layer inserted
# -------------------------

class SiameseNetQuantum(nn.Module):
    """
    Siamese network with quantum embedding layer.:
      return similarity, embedding1, embedding2, distance
    Comparator: squared diff -> Linear(embedding_size,1) -> Sigmoid (unchanged).
    """
    def __init__(self, input_dims=12, embedding_size=128, n_qubits=7, n_layers=1, use_quantum=True):
        super().__init__()
        self.use_quantum = use_quantum
        self.features = FeatureNet(input_dims=input_dims, embedding_size=embedding_size)
        if use_quantum:
            self.qblock = QuantumBlock(embedding_size=embedding_size, n_qubits=n_qubits, n_layers=n_layers)

        self.final_fc = nn.Sequential(nn.Linear(embedding_size, 1), nn.Sigmoid())

    def _embed_branch(self, x):
        # x: [B, T, F]
        e = self.features(x)          # [B, embedding_size]
        if self.use_quantum:
            e = self.qblock(e)                 # [B, embedding_size]
        return e

    def forward(self, x1, x2):
        emb1 = self._embed_branch(x1)
        emb2 = self._embed_branch(x2)


        distance = F.pairwise_distance(emb1, emb2)
        
        similarity = self.final_fc((emb1 - emb2) ** 2  )                 # [B, 1] (same as your code)

        return similarity, distance, emb1, emb2  # return all outputs 












