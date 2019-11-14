from utils import *         # File with miscelaneous helper methods

print("START")
create_names_with_batch("./data/train",3,"./names_log",30, title ='30_')

create_names_with_batch("./data/train",3,"./names_log",300,  title ='300_')

create_names_with_batch("./data/train",3,"./names_log",1500,  title ='1500_')

create_names_with_batch("./data/train",3,"./names_log",3000,  title = '3000_')
print("END")