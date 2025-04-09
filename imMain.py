import argparse
import os
from ImpossibleOne import SophisticatedModel
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from impossibleTrain import TrainerModel
from pytorch_lightning.plugins import DDPPlugin
import glob

cudnn.benchmark = True
import torch_geometric
import sys
# sys.path.insert(0, "../")
from v1.main import KFOLD


def load_dataset(pts, args):
    all_files = glob.glob(f"{args.graph_path}/{args.fold}/*.pt")

    selected_files = []

    for i in all_files:
        for j in pts:
            if i.endswith(str(j) + ".pt") and "dgl" not in i:
                graph = torch.load(i)
                selected_files.append(graph)
    return selected_files


def main(args):
    cwd = os.getcwd()

    def write(director, name, *string):
        string = [str(i) for i in string]
        string = " ".join(string)
        with open(os.path.join(director, name), "a") as f:
            f.write(string + "\n")

    store_dir = os.path.join(args.output, str(args.fold))
    print = partial(write, cwd, args.output + "/" + "log_f" + str(args.fold))

    os.makedirs(store_dir, exist_ok=True)

    print(args)

    train_patient, test_patient = KFOLD[args.fold]

    train_dataset = load_dataset(train_patient, args)
    test_dataset = load_dataset(test_patient, args)

    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=1,
    )

    test_loader = torch_geometric.loader.DataLoader(
        test_dataset,
        batch_size=1,
    )

    model = SophisticatedModel(num_layers=args.num_layers, mdim=args.mdim,
                               global_embed=args.global_embed, edge_embed=args.edge_embed)
    CONFIG = collections.namedtuple('CONFIG',
                                    ['lr', 'logfun', 'verbose_step', 'weight_decay', 'store_dir', "fold"])
    config = CONFIG(args.lr, print, args.verbose_step, args.weight_decay, store_dir, args.fold)

    model = TrainerModel(config, model)

    plt = pl.Trainer(max_epochs=args.epoch, num_nodes=1, val_check_interval=args.val_interval, checkpoint_callback=False,
                     gpus=1,
                     strategy=DDPPlugin(find_unused_parameters=True), logger=False)
    plt.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=80, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--acce", default="ddp", type=str)
    parser.add_argument("--val_interval", default=0.8, type=float)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--verbose_step", default=10, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--mdim", default=2048, type=int)
    parser.add_argument("--output", default="results/edge_node", type=str)  #####################################
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--graph_path", default="edge_node_graphs", type=str)  ###############################
    parser.add_argument("--global_embed", default=False, type=bool)  ##########
    parser.add_argument("--edge_embed", default=False, type=bool)  ########

    args = parser.parse_args()
    # for f in range(3):
    #     args.fold = f
    #     main(args)
    main(args)

