import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argument
import data_loader
from model.gcn import GNN
import utils

if __name__ == '__main__':
    parser = argument.load_parser()

    # TODO: arguments for boundary nodes
    args = parser.parse_args()

    data = data_loader.load(args)
    device = utils.get_device(args)

    surrogate = GNN(args, data.num_features, data.num_classes, surrogate=True)
    surrogate.train(data, device)
    _, _, posterior, Z = surrogate.predict(data, device, return_posterior=True, return_logit=True)

    boundary_scores = []
    for p in posterior:
        boundary_scores.append(utils.boundary_score(p))

    plt.figure()
    sns.set_theme()
    ax = sns.histplot(x = boundary_scores, bins=10)
    ax.set_xlabel('Boundary score')
    plt.show()
    