import argument
import data_loader
import utils
from model.gnn import GNN

if __name__ == '__main__':
    parser = argument.load_parser()
    args = parser.parse_args()

    device = utils.get_device(args)
    data = data_loader.load(args)

    model = GNN(args, data.num_features, data.num_classes, surrogate=False)
    model.train(data, device)
    result = model.evaluate(data, device)

    print('*' * 50, 'Result', '*' * 50)
    print(result)
    print('*' * 100)