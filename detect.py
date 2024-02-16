from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


dog_mappings = {
    0: '絲毛梗',
    1: '蘇格蘭鹿犬',
    2: '切薩皮克灣檢出犬',
    3: '伊比薩獵犬',
    4: '硬毛狐獾梗',
    5: '薩魯基犬',
    6: '可卡獵犬',
    7: '司基爾比克犬',
    8: '波爾索伊犬',
    9: '彭布羅克威爾斯柯基犬',
    10: '庫瓦茲犬',
    11: '斯塔福郡牛頭梗',
    12: '標準貴賓犬',
    13: '愛斯基摩狗',
    14: '英國獵狐犬',
    15: '金毛獵犬',
    16: '西利漢梗',
    17: '日本狆',
    18: '迷你雪納瑞',
    19: '馬拉穆特犬',
    20: '馬林犬',
    21: '北京狗',
    22: '巨型雪納瑞',
    23: '墨西哥無毛犬',
    24: '杜賓犬',
    25: '中型雪納瑞',
    26: '阿比亞獵犬',
    27: '德國牧羊犬',
    28: '佛蘭德斯牧牛犬',
    29: '西伯利亞哈士奇',
    30: '諾里奇梗',
    31: '愛爾蘭梗',
    32: '諾福克梗',
    33: '聖伯納犬',
    34: '波士頓梗',
    35: '布里亞德犬',
    36: '西藏獒犬',
    37: '牛頭獒犬',
    38: '馬爾濟斯犬',
    39: '凱利藍梗',
    40: '庫瓦茲犬',
    41: '大瑞士山地犬',
    42: '萊克蘭梗',
    43: '布倫漢姆獵犬',
    44: '巴吉特犬',
    45: '西高地白梗',
    46: '吉娃娃',
    47: '邊境牧羊犬',
    48: '瑞德骨',
    49: '愛爾蘭獵狼犬',
    50: '布魯塔克犬',
    51: '迷你貴賓犬',
    52: '卡迪根威爾斯柯基犬',
    53: '恩特勒布赫獵犬',
    54: '挪威獵鹿犬',
    55: '德國短毛指示犬',
    56: '伯恩山犬',
    57: '蝴蝶犬',
    58: '西藏梗',
    59: '戈登塞特犬',
    60: '美國斯塔福郡梗',
    61: '維茲拉犬',
    62: '牧羊犬',
    63: '威瑪犬',
    64: '迷你品犬',
    65: '拳擊手犬',
    66: '鬆獅犬',
    67: '老英國牧羊犬',
    68: '巴哥犬',
    69: '羅德西亞脊背犬',
    70: '蘇格蘭梗',
    71: '獅子狗',
    72: '猴頭梗',
    73: '威帕犬',
    74: '蘇塞克斯獵犬',
    75: '水獺犬',
    76: '扁平毛獵犬',
    77: '英格蘭雪達犬',
    78: '意大利灰狗',
    79: '拉布拉多獵犬',
    80: '科利犬',
    81: '開恩梗',
    82: '羅威納犬',
    83: '澳洲梗',
    84: '玩具梗',
    85: '喜瑞都牧羊犬',
    86: '非洲獵犬',
    87: '紐芬蘭犬',
    88: '獵犬',
    89: '拉薩犬',
    90: '比格犬',
    91: '薩摩耶犬',
    92: '大丹犬',
    93: '艾爾戴爾梗',
    94: '血獺犬',
    95: '愛爾蘭雪達犬',
    96: '凱斯犬',
    97: '丹迪丁蒙特犬',
    98: '巴辛吉犬',
    99: '貝德靈頓梗',
    100: '阿彭策勒',
    101: '克倫伯犬',
    102: '玩具貴賓犬',
    103: '大比利牛斯犬',
    104: '英國斯普林格犬',
    105: '阿富汗獵犬',
    106: '布列塔尼犬',
    107: '威爾斯斯普林格犬',
    108: '波士頓斗牛犬',
    109: '野狗',
    110: '柔毛小麥犬',
    111: '捲毛獵犬',
    112: '法國斗牛犬',
    113: '愛爾蘭水獺犬',
    114: '博美犬',
    115: '布拉班康獅子犬',
    116: '約克夏梗',
    117: '黑鬃犬',
    118: '萊茵博格犬',
    119: '黑褐獵浣熊犬'
}
model = YOLO('best.pt')
img = Image.open('dog3.JPG')
# Load the trained model for inference
inference_model = YOLO("best.pt")
results = model.predict(source=img, imgsz=640)

result = results[0]
bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
classes = np.array(result.boxes.cls.cpu(), dtype="int")
confidences = np.array(result.boxes.conf.cpu(), dtype="float")

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
chinese_font_name =  "STHeiti Medium.ttc"
font = ImageFont.truetype(chinese_font_name, 50)
draw = ImageDraw.Draw(img)

total = 0
for cls, bbox, conf in zip(classes, bboxes, confidences):
    (x, y, x2, y2) = bbox
    total +=1
    label = f"{dog_mappings[cls]}: {conf:.2f}"
    draw.rectangle([(x, y), (x2, y2)], outline=(0, 255, 0), width=3)
    draw.text((x, y - 80), label, font=font, fill=(255,0,0))

plt.imshow(img)
plt.savefig('dog3_finish.JPG')
