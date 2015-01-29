# TJC
# Extract only the words from the training set for use with Brown clustering
# Words must be separated by whitespace for Brown clustering algorithm

output_file = open("cluster_text.txt", "w")

with open("train.gold") as f:
    for line in f:
        if len(line) > 1:
            output_file.write(line.split()[1] + " ")
            
output_file.close()