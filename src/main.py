from a2c_torch import A2C 
import argparse
import sys 
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default="NEL-v0")
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str, default = "")
    parser.add_argument('--lr',dest='lr',type=float, default = 1e-3)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    # You want to create an instance of the DQN_Agent class here, and then train / test it. 
    agent = A2C( args.env , args.lr, render = args.render )
    if args.model_file != "":
        agent.load_model( args.model_file )
    if args.train:
        agent.train()
    else:
        agent.test()

if __name__ == '__main__':
    main( sys.argv )