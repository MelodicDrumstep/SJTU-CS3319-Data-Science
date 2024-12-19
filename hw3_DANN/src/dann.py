import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_dims=[128, 64]):
        super(FeatureExtractor, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LabelClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=3):
        super(LabelClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[32], output_dim=2):
        super(DomainClassifier, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features, alpha=1.0):
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.network(reversed_features)

class DANNModel(nn.Module):
    def __init__(self, input_dim=310, feature_hidden_dims=[128, 64], label_output_dim=3,
                 domain_hidden_dims=[32], mode="DA", use_domain_classifier=True):
        """
        DANN Model.
        :param mode: "DA" or "DG"
        """
        super(DANNModel, self).__init__()
        self.mode = mode
        self.use_domain_classifier = use_domain_classifier

        # Feature Extractor
        self.feature_extractor = FeatureExtractor(input_dim, feature_hidden_dims)

        # Label Classifier
        self.label_classifier = LabelClassifier(feature_hidden_dims[-1], label_output_dim)

        # Domain Classifier
        if use_domain_classifier:
            domain_output_dim = 2 if mode == "DA" else 11  # Binary for DA, 11 for DG
            logging.debug(f"[DANNModel::__init__] domain_output_dim is {domain_output_dim}")
            self.domain_classifier = DomainClassifier(feature_hidden_dims[-1], domain_hidden_dims, domain_output_dim)
        else:
            self.domain_classifier = None

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        label_preds = self.label_classifier(features)

        if self.use_domain_classifier:
            reversed_features = GradientReversalLayer.apply(features, alpha)
            domain_preds = self.domain_classifier(reversed_features)
            return label_preds, domain_preds
        return label_preds
