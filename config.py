
def read_config():
    ConfigFilePath = 'config_train'
    config_info = {}
    with open(ConfigFilePath) as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.rstrip('\r\n')
            item = line.split('=')
            key = item[0]
            if key == 'WRFFileDir':
                config_info[key] = item[1]
            elif key == 'TruthFileDir':
                config_info[key] = item[1]
            elif key == 'TruthFileDirGrid':
                config_info[key] = item[1]
            elif key == 'VisResultFileDir':
                config_info[key] = item[1]
            elif key == 'GraphFileDir':
                config_info[key] = item[1]
            elif key == 'ModelFileDir':
                config_info[key] = item[1]
            elif key == 'RecordFileDir':
                config_info[key] = item[1]
            elif key == 'GridRowColNum':
                config_info[key] = int(item[1])
            elif key == 'ForecastHourNum':
                config_info[key] = int(item[1])
            elif key == 'TruthHistoryHourNum':
                config_info[key] = int(item[1])
            elif key == 'GraphChooseLayers':
                l_list = []
                for c in item[1].split(','):
                    l_list.append(int(c))
                config_info[key] = l_list
            elif key == 'ScanStartTime':
                config_info[key] = item[1]
            elif key == 'ScanEndTime':
                config_info[key] = item[1]
            elif key == 'NetName':
                config_info[key] = item[1]
            elif key == 'Batchsize':
                config_info[key] = int(item[1])
            elif key == 'LearningRate':
                config_info[key] = float(item[1])
            elif key == 'EpochNum':
                config_info[key] = int(item[1])
            elif key == 'WRFChannelNum':
                config_info[key] = int(item[1])
            elif key == 'Device':
                config_info[key] = item[1]
            else:
                print('no this item: {}'.format(key))
                assert False
    return config_info

if __name__ == "__main__":
    t = read_config()
    print(t)



