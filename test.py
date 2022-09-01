import os
import yaml
# from enum import Enum

# class hand_class(Enum):
#     好 = 0
#     你 = 1    

record_class = '好'
record_start_idx = 0 

yaml_path = os.path.join('index_data.yaml') 

# def read_yaml_all():
#     try:
#         with open(yaml_path,"r",encoding="utf-8") as f:
#             data=yaml.load(f,Loader=yaml.FullLoader)
#             return data['data_index']['好']
#     except:
#         return None

    
# def write_yaml(r):
#     with open(yaml_path, "w", encoding="utf-8") as f:
#         yaml.dump(r,f)


def update_data_index(hand_class, alter_idx = 10):
    try:
        # 读取
        with open(yaml_path,"r",encoding="utf-8") as f:
            data=yaml.load(f,Loader=yaml.FullLoader)
            # 修改
            with open(yaml_path, "w", encoding="utf-8") as s:
                if alter_idx == 0:
                    data['data_index'][hand_class] += 1
                elif alter_idx > 0:
                    data['data_index'][hand_class] = alter_idx
                yaml.dump(data,s,allow_unicode=True)
            return data
    except Exception as e:
        print("访问失败: ",{e})
        return None

update_data_index('好')

