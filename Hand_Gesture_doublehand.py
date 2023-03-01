'''
参考:https://github.com/shoujou/Hand-Gestures
'''

import tkinter as tk
import cv2
import time
import random

'''module'''
from HandGesture_module import ai as gestureai
from HandGesture_module import cv as gesturecv
from torchvision import transforms
import torch
import glob
import os
''''''''''''

# ウィンドウサイズの設定
WINDOW_WIDTH  = 300
WINDOW_HEIGHT = 300

# モデルに入力する画像のサイズを設定
INPUT_WIDTH  = 50
INPUT_WIDTH  = 50

# デモアプリを作成
class Demo(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        
        # 各パラメータ設定
        self.xmin_l, self.xmax_l = 175, 325  # 左手の枠 x
        self.ymin_l, self.ymax_l = 200, 350  # 左手の枠 y
        self.xmin_r, self.xmax_r = 325, 475  # 右手の枠 x
        self.ymin_r, self.ymax_r = 200, 350  # 右手の枠 y
        
        # カメラ設定
        #環境のカメラによって引数を変更する
        self.capture = cv2.VideoCapture(2)
 
        # ウィンドウの設定
        self.master.title('PBLデモ')
        self.master.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')
 
        # 実行内容
        self.pack()
        self.create_widget()
    
    # ==========
    # create_widgetメソッドを定義
    # ==========
    def create_widget(self):
 
        # ウィジェット変数を生成
        self.msg_machine = tk.StringVar()
        self.msg_machine.set('')
        self.msg_good = tk.StringVar()
        self.msg_good.set('')
        self.msg_bad = tk.StringVar()
        self.msg_bad.set('')
        self.msg_all = tk.StringVar()
        self.msg_all.set('')
 
        # label1ウィジェットを作成(作業開始, 終了キーを表示)
        self.label1 = tk.Label(self.master, text='作業開始:Sキー\n作業終了:Qキー', font=('MSゴシック', '24', 'bold'))
        self.label1.bind('<s>', self.do_process)
        self.label1.pack()
        self.label1.focus_set()
        
        # label2ウィジェットを作成 (機械名表示)
        self.label2 = tk.Label(self.master, textvariable=self.msg_machine, font=('MSゴシック', '24', 'bold'))
        self.label2.pack()
        
        # label3ウィジェットを作成 (良品数表示)
        self.label3 = tk.Label(self.master, textvariable=self.msg_good, font=('MSゴシック', '24', 'bold'))
        self.label3.pack()
        
        # label4ウィジェットを作成 (不良品数表示)
        self.label4 = tk.Label(self.master, textvariable=self.msg_bad, font=('MSゴシック', '24', 'bold'))
        self.label4.pack()
        
        # label4ウィジェットを作成 (全体数表示)
        self.label5 = tk.Label(self.master, textvariable=self.msg_all, font=('MSゴシック', '24', 'bold'))
        self.label5.pack()
    
    # ==========
    # 作業メソッドを定義
    # ==========
    def do_process(self, event):



        model, input_size = gestureai.initialize_model(model_name="resnet", num_classes=7, feature_extract=False, use_pretrained=True, binary=True)
        #model_path = "./ex3.pth"
        model_path = "./model/mix/model_weights.pth"
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # data = "image"
        # classification_allcsv_fullpath = 'test.csv'
        # imgfullpaths = ['left.jpg', 'right.jpg']
        # gestureai.imgfullpaths2allcsv(imgfullpaths, classification_allcsv_fullpath)
        # class_names=['0','1','2','3','4','5','g']

        '''下記のうち関数の入出力を変更'''



        # 作業1 : 機械名選択
        print('機械名を選択してください')
        machine_name = self.select_machine('機械名', model, input_size)
        
        # 問題なければgoodで次の作業へ移る
        
        # 作業2 : 良品数入力
        print('良品数を入力してください')
        good_n = self.input_numbers('良品数', model, input_size)
        
        # 問題なければgoodで次の作業へ移る
        
        # 作業3 : 不良品数入力
        print('不良品数を入力してください')
        bad_n = self.input_numbers('不良品数', model, input_size)
        
        # 問題なければgoodで次の作業へ移る
        
        # 作業4 : ファイルに書き込み
        all_n = int(good_n)+int(bad_n)
        print(f'機械名:{machine_name}, 良品数:{good_n}, 不良品数:{bad_n}, 全体数:{all_n}')
        self.msg_all.set(f'全体数:{all_n}')
        self.label5.update()
        
        #self.master.destroy()
    
    # ==========
    # 作業1 : 機械名選択
    # ==========
    def select_machine(self, type, model, input_size):
        machine_name = ''
        i = 0
        while(True):
            i = i + 0.1
            frame = self.get_img()
            cv2.imshow('capture',frame)
            if i > 10:
                machine_name = machine_name + str(self.gesture_classification(frame, model, input_size))
            
                self.msg_machine.set(f'{type}:{machine_name}')
                self.label2.update()
                #time.sleep(3)
                i = 0
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return machine_name
        
    # ==========
    # 作業2,3 : 良品数・不良品数入力
    # ==========
    def input_numbers(self, type, model, input_size):
        num = ''
        i = 0
        while(True):
            i = i + 0.1
            frame = self.get_img()
            cv2.imshow('capture',frame)
            if i > 10:
                num = num + str(self.gesture_classification(frame, model, input_size))
            
                if type == '良品数':
                    self.msg_good.set(f'{type}:{num}')
                    self.label3.update()
                else:
                    self.msg_bad.set(f'{type}:{num}')
                    self.label4.update()
                #time.sleep(3)
                i = 0
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return num
    
    # ==========
    # 作業4 : ファイル書き込み
    # ==========
    def write_file(self):
        
        return 0
    
    # ==========
    # ジェスチャーを識別
    # ==========
    def gesture_classification(self, frame, model, input_size):
        dataframe_l = frame[self.ymin_l:self.ymax_l, self.xmin_l:self.xmax_l]  # 左手の指定範囲の画像を取得
        dataframe_r = frame[self.ymin_r:self.ymax_r, self.xmin_r:self.xmax_r]  # 右手の指定範囲の画像を取得
        self.save_img(dataframe_l, dataframe_r)
        
        ''' ジェスチャー識別コード ここから '''
        # input input_size model


        classification_allcsv_fullpath = './test_doublehand.csv'
        data = "image"
        data_transforms = {
            'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            ]),
        } 
        dataset = gestureai.only_img_Datasets(classification_allcsv_fullpath, data, data_transforms['val'])
        dataloaders = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        AI_result = gestureai.classio(model, dataloaders)
        print(AI_result[0][0])
        print(AI_result[1][0])
        class_names=['0','1','2','3','4','5','g']
        output_names=['0','1','2','3','4','5','6','7','8','9','10','good']
        # output = AI_result[0][0] + AI_result[0][1]


        result = AI_result[0][0] + AI_result[1][0]
        # result = str(random.randint(0,9))
        ''' ジェスチャー識別コード ここまで '''
               
        return result
    
    # ==========
    # 画像を取得
    # ==========
    def get_img(self):
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (self.xmin_l-2, self.ymin_l-2), (self.xmax_l+2, self.ymax_l+2), (0, 0, 255), 2)  # 左手の指定範囲に赤枠
        cv2.rectangle(frame, (self.xmin_r-2, self.ymin_r-2), (self.xmax_r+2, self.ymax_r+2), (0, 0, 255), 2)  # 右手の指定範囲に赤枠
        return frame

    # ==========
    # 画像を保存
    # ==========    
    def save_img(self, dataframe_l, dataframe_r):
        # 2値化
        dataframe_l = self.thresholding_img(dataframe_l)
        dataframe_r = self.thresholding_img(dataframe_r)
        
        # リサイズ
        dataframe_l = self.resize_img(dataframe_l)
        dataframe_r = self.resize_img(dataframe_r)
        
        cv2.imwrite('image/left.jpg', dataframe_l)
        cv2.imwrite('image/right.jpg', dataframe_r)
    
    # ==========
    # 画像を2値化
    # ==========
    def thresholding_img(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        th, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image_gray
    
    # ==========
    # 画像をリサイズ
    # ==========
    def resize_img(self, image):
        image_resize = cv2.resize(image, dsize=(INPUT_WIDTH, INPUT_WIDTH))
        return image_resize
    

def main():
    
    '''
    1. 作業開始ボタン押す
    2. 作業開始から3秒経過
    3. 静止画像取得
    4. モデルに静止画像入力
    5. モデルの出力結果を表示
    6. 作業終了ボタンが押されるまで、2-5を繰り返す
    '''
    
    root = tk.Tk()
    app = Demo(master = root)
    
    # メインループ
    app.mainloop()

if __name__ == '__main__':
    main()