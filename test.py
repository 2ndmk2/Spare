
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f",
                    help="name of folder including result")
args = parser.parse_args()

print (args.f+1)

file = open(args.f,"w")
file.write("test")
file.close()