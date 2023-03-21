import datetime

def write_log(s):
    output = 'logs/test.txt'
    currentDT = datetime.datetime.now()
    # print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
    print(s)
    file = open(output, 'a')
    file.write("{}\n".format(s))
    file.close()