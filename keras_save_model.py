from keras.models import load_model
from keras.models import model_from_json, model_from_yaml
# 模型和权重
model.save('my_model.h5')
model = load_model('my_model.h5')
# 模型
json_string = model.to_json()
model = model_from_json(json_string)
yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
# 权重
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5', by_name=True)
