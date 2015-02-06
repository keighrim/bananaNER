# TJC
# Extract only the words from the training/test/dev set for use with Brown 
# clustering. Words must be separated by whitespace for clustering algorithm

output_file = open("cluster_text.txt", "w")

def extract_text(filename):
    with open(filename) as f:
        for line in f:
            if len(line) > 1:
                output_file.write(line.split()[1] + " ")
            
extract_text("../dataset/train.gold")
extract_text("../dataset/dev.gold")
extract_text("../dataset/test.gold")

output_file.close()