import pandas as pd
import model.model_pathloss as model_pathloss
import tool.metrics as metrics

data = pd.read_csv('../dataset/v1/v1_custom/pathloss_v1_valid_cs.csv')
fspl_path_loss_line = []
for index, row in data.iterrows():
    row = row.tolist()
    row.append(model_pathloss.get_distance_with_rssi_fspl(row[2]))
    row.append(model_pathloss.get_distance_with_rssi(row[2], 5, 5))
    row.append(model_pathloss.log_distance(row[2]))
    fspl_path_loss_line.append(row)

result = pd.DataFrame(fspl_path_loss_line)

metrics.show_all_metrics_result(result[2], result[1], result[12])

