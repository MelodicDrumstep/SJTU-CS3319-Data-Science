import torch
import torch.nn as nn
import torch.optim as optim

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_dims=[128, 64]):
        """
        Feature Extractor
        :param input_dim: Dimension of the input features (default 310)
        :param hidden_dims: List of hidden layer dimensions (default [128, 64])
        """
        super(FeatureExtractor, self).__init__()
        layers = []
        # Build a multi-layer fully connected network
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

class LabelClassifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=3):
        """
        Label Classifier
        :param input_dim: Dimension of the input features (default 64)
        :param output_dim: Number of output classes (default 3)
        """
        super(LabelClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[32], output_dim=1):
        """
        Domain Classifier
        :param input_dim: Dimension of the input features (default 64)
        :param hidden_dims: List of hidden layer dimensions (default [32])
        :param output_dim: Output dimension (default 1)
        """
        super(DomainClassifier, self).__init__()
        layers = []
        # Build a multi-layer fully connected network
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x, alpha):
        x = self.fc_layers(x)
        x = x * alpha  # Gradient reversal layer
        return x

class DANNModel(nn.Module):
    def __init__(self, input_dim=310, feature_hidden_dims=[128, 64], 
                 label_output_dim=3, domain_hidden_dims=[32], domain_output_dim=1,
                 use_domain_classifier=True):
        """
        DANN Model
        :param input_dim: Dimension of the input features (default 310)
        :param feature_hidden_dims: List of hidden layer dimensions for the feature extractor (default [128, 64])
        :param label_output_dim: Number of output classes for the label classifier (default 3)
        :param domain_hidden_dims: List of hidden layer dimensions for the domain classifier (default [32])
        :param domain_output_dim: Output dimension for the domain classifier (default 1)
        :param use_domain_classifier: Whether to use the domain classifier (default True)
        """
        super(DANNModel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, feature_hidden_dims)
        self.label_classifier = LabelClassifier(feature_hidden_dims[-1], label_output_dim)
        
        if use_domain_classifier:
            self.domain_classifier = DomainClassifier(feature_hidden_dims[-1], domain_hidden_dims, domain_output_dim)
        else:
            self.domain_classifier = None

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        label_pred = self.label_classifier(features)
        
        if self.domain_classifier is not None:
            domain_pred = self.domain_classifier(features, alpha)
            return label_pred, domain_pred
        else:
            return label_pred