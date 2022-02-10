import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--hello', type=int)
parser.add_argument('--how', type=int)
args = parser.parse_args(["--hello", "1", "--how", "2"])
print(args.hello, args.how, args.hello + args.how)