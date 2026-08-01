# Severity Classification
> solely for SAIG audition <br>

utilizing features like **time_period, agency, province, first_vehicle, cause, accident_type, weather, road_type, road_slope** to predict severity of the accident
# The Evaluation
```
              precision    recall  f1-score   support

           0       0.66      0.63      0.64      9769
           1       0.53      0.27      0.36      8537
           2       0.11      0.27      0.16      1205
           3       0.25      0.44      0.32      1436
           4       0.08      0.16      0.11       705
           5       0.04      0.17      0.06       322

    accuracy                           0.44     21974
   macro avg       0.28      0.32      0.27     21974
weighted avg       0.52      0.44      0.46     21974

Macro F1: 0.2739791812736472
```
# Comments
This was very fun experience and showed how much more I can learn in fields of machine learning. Like there are so much more room for improvement. I would love to study more deeply into every topics of machine learning cause I was so fascinated by how the algorythms are working and also cleaning, EDA the data is such a great visualization of how data can be use to the greater potential. 

# Try
```
# install dependency
!pip install category_encoders
# import library
import requests
import joblib
import io

# model class definition
class AccidentSeverityModel:
    REQUIRED_FIELDS = [
        'time_period', 'province', 'agency', 'first_vehicle',
        'cause', 'accident_type', 'weather', 'road_type', 'road_slope'
    ]

    def __init__(self, model, target_encoder, freq_maps,
                 time_dummy_columns, feature_columns, severity_labels):
        self.model = model
        self.target_encoder = target_encoder
        self.freq_maps = freq_maps
        self.time_dummy_columns = time_dummy_columns
        self.feature_columns = feature_columns
        self.severity_labels = severity_labels

    def _preprocess(self, **kwargs):
        missing = [f for f in self.REQUIRED_FIELDS if f not in kwargs]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        import pandas as pd
        row = pd.DataFrame([kwargs])

        row = pd.get_dummies(row, columns=['time_period'], prefix='time')
        for col in self.time_dummy_columns:
            if col not in row.columns:
                row[col] = 0

        for col, freq_map in self.freq_maps.items():
            row[col + '_freq'] = row[col].map(freq_map).fillna(0)
            row = row.drop(columns=[col])

        target_cols = list(self.target_encoder.cols)
        row[target_cols] = self.target_encoder.transform(row[target_cols])

        row = row.reindex(columns=self.feature_columns, fill_value=0)
        return row

    def predict(self, return_label=True, **kwargs):
        X_row = self._preprocess(**kwargs)
        pred = self.model.predict(X_row)[0]
        return self.severity_labels[pred] if return_label else pred

    def predict_proba(self, **kwargs):
        X_row = self._preprocess(**kwargs)
        proba = self.model.predict_proba(X_row)[0]
        return {self.severity_labels[i]: p for i, p in enumerate(proba)}


# download model
url = "https://raw.githubusercontent.com/oo90x/SAIGdiwa/main/severity_model.pkl"
response = requests.get(url)
response.raise_for_status()

severity_model = joblib.load(io.BytesIO(response.content))
print("พร้อมแล้วพร้อมแล้วตื่นเต้นตื่นเต้น><!")
```
## Cheat Sheet

> === time_period (5 unique values) === <br>
['Afternoon', 'Dawn', 'Morning', 'Night', 'Dusk'] <br>

=== province (77 unique values) === <br>
['ประจวบคีรีขันธ์', 'พังงา', 'เพชรบูรณ์', 'ชัยนาท', 'สุราษฎร์ธานี', 'ยโสธร', 'นครสวรรค์', 'นครพนม', 'ลพบุรี', 'ปทุมธานี', 'สุพรรณบุรี', 'ชุมพร', 'ขอนแก่น', 'นครราชสีมา', 'พัทลุง', 'อุตรดิตถ์', 'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'น่าน', 'สกลนคร', 'กำแพงเพชร', 'แพร่', 'อ่างทอง', 'อยุธยา', 'อุดรธานี', 'ลำปาง', 'สระบุรี', 'นครปฐม', 'พิษณุโลก', 'สระแก้ว', 'สิงห์บุรี', 'ลำพูน', 'กรุงเทพมหานคร', 'อุบลราชธานี', 'ตาก', 'บึงกาฬ', 'ปราจีนบุรี', 'สงขลา', 'สมุทรปราการ', 'นครศรีธรรมราช', 'มุกดาหาร', 'สมุทรสาคร', 'ศรีสะเกษ', 'ระยอง', 'เพชรบุรี', 'มหาสารคาม', 'ยะลา', 'ร้อยเอ็ด', 'กระบี่', 'สุโขทัย', 'สมุทรสงคราม', 'ตรัง', 'เชียงราย', 'เชียงใหม่', 'นนทบุรี', 'นราธิวาส', 'ภูเก็ต', 'กาฬสินธุ์', 'อุทัยธานี', 'กาญจนบุรี', 'ปัตตานี', 'เลย', 'สตูล', 'สุรินทร์', 'อำนาจเจริญ', 'ราชบุรี', 'ระนอง', 'แม่ฮ่องสอน', 'ตราด', 'พะเยา', 'พิจิตร', 'หนองบัวลำภู', 'บุรีรัมย์', 'หนองคาย', 'ชัยภูมิ', 'นครนายก'] <br>

=== **agency** (3 unique values) === <br>
['กรมทางหลวง', 'กรมทางหลวงชนบท', 'การทางพิเศษแห่งประเทศไทย'] <br>

=== **first_vehicle** (13 unique values) === <br>
['รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ', 'รถบรรทุก 6 ล้อ', 'รถปิคอัพบรรทุก 4 ล้อ', 'รถจักรยานยนต์', 'อื่นๆ', 'รถบรรทุกมากกว่า 6 ล้อ ไม่เกิน 10 ล้อ', 'รถบรรทุกมากกว่า 10 ล้อ (รถพ่วง)', 'รถตู้', 'คนเดินเท้า', 'รถสามล้อเครื่อง', 'รถปิคอัพโดยสาร', 'รถโดยสารขนาดใหญ่', 'รถจักรยาน'] <br>

=== **cause** (18 unique values) === <br>
['หลับใน', 'ฝ่าฝืนกฎจราจร', 'ตัดหน้า/หยุดรถกระทันหัน', 'อุปกรณ์ชำรุด', 'ถนนมีปัญหา', 'ไม่ระบุ', 'ประมาท', 'ใช้สารออกฤทธิ์ต่อจิตและประสาท', 'มีกองวัสดุ/สิ่งกีดขวาง', 'สูญเสียการควบคุม', 'ทางโค้งอันตราย', 'อาการป่วย/โรคประจำตัว', 'ไม่คุ้นชิน', 'อื่นๆ', 'ไม่ให้สัญญาณ', 'ปัญหาด้านทัศนวิสัย', 'มีสิ่งกีดขวางบนทางหลวง (วัสดุหรือสัตว์)', 'มีสิ่งรบกวน'] <br>

=== **accident_type** (9 unique values) === <br>
['พลิกคว่ำ/ตกถนนในทางตรง', 'ชน', 'อื่นๆ', 'พลิกคว่ำ/ตกถนนในทางโค้ง', 'เสียหลัก', 'เสียการควบคุม', 'ไม่มีรายละเอียด', 'ตกจากรถ', 'เสียการควบคุมในทางโค้ง'] <br>

=== **weather** (3 unique values) ===  <br>
['แจ่มใส', 'ฝนตก', 'อื่นๆ'] <br>

=== **road_type** (30 unique values) === <br>
['ทางตรง', 'ทางโค้งกว้าง', 'ทางเชื่อมเข้าบริเวณหน้าโรงเรียน', 'ทางแยกรูปตัวบวก', 'ทางโค้งปกติ', 'ทางเชื่อมเข้าพื้นที่สาธารณะ/เชิงพาณิชย์', 'ทางแยกรูปตัว T', 'ทางแยกต่างระดับ / Ramps', 'ทางสามแยก (T)', 'เข้าพื้นที่ส่วนบุคคล', 'ทางสามแยก (Y)', 'ทางเชื่อมเข้าพื้นที่ส่วนบุคคล', 'ทางโค้งหักศอก', 'ทางลอด', 'ทางสี่แยก', 'ทางแยกรูปตัว Y', 'วงเวียน', 'มีการเปลี่ยนจำนวนช่องจราจร', 'สะพาน', 'จุดกลับรถต่างระดับ', 'ทางม้าลาย', 'สถานี/จุดขึ้นลงระบบขนส่งสาธารณะ', 'ทางร่วม', 'เข้าสถานศึกษา / สถานที่ราชการ', 'ทางคนเดินเท้า', 'ทางรถไฟตัดผ่าน', 'ทางรถจักรยานยนต์', 'ทางยกระดับ', 'ทางจักรยาน', 'อื่นๆ'] <br>

=== **road_slope** (3 unique values) === <br>
['ที่ราบ/ไม่มีความลาดชัน', 'ไม่ระบุ', 'ทางลาดชัน'] <br>

```
# use
result = severity_model.predict(
    time_period='',
    province='',
    agency='',
    first_vehicle='',
    cause='',
    accident_type='',
    weather='',
    road_type='',
    road_slope=''
)
print(f"คำตอบคืออ อ: {result}")
```
