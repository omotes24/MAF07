import torch


class CWKNNMean:
    """Class-wise kNN mean distance on raw features."""

    def __init__(self, k=150, batch_size=256, device="cuda"):
        self.k = k
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def fit(self, train_features, train_labels):
        x = torch.as_tensor(
            train_features, dtype=torch.float32, device=self.device
        )
        y = torch.as_tensor(train_labels, device=self.device)
        self.classes = torch.unique(y, sorted=True)
        self.banks = [x[y == c].contiguous() for c in self.classes]
        self.bank_norms = [(bank * bank).sum(1) for bank in self.banks]
        return self

    @torch.no_grad()
    def ood_score(self, features):
        """Larger values indicate more OOD-like samples."""
        query = torch.as_tensor(
            features, dtype=torch.float32, device=self.device
        )
        chunks = []
        for start in range(0, len(query), self.batch_size):
            q = query[start : start + self.batch_size]
            q_norm = (q * q).sum(1, keepdim=True)
            best = torch.full(
                (len(q),), torch.inf, dtype=q.dtype, device=q.device
            )
            for bank, bank_norm in zip(self.banks, self.bank_norms):
                distance2 = (
                    q_norm + bank_norm[None, :] - 2.0 * q @ bank.T
                ).clamp_min(0.0)
                k_c = min(self.k, len(bank))
                class_distance = torch.sqrt(
                    torch.topk(distance2, k_c, largest=False).values
                ).mean(1)
                best = torch.minimum(best, class_distance)
            chunks.append(best)
        return torch.cat(chunks).cpu().numpy()

    def id_score(self, features):
        """Larger values indicate more ID-like samples."""
        return -self.ood_score(features)
