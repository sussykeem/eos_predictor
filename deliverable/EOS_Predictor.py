import sys
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QPalette, QImage
from PyQt5.QtCore import Qt, QProcess
from PIL import Image
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, rdDepictor
from PyQt5.QtWidgets import ( 
    QApplication, QMainWindow, 
    QLabel, QWidget, QPushButton, QTextEdit, QLineEdit,
    QVBoxLayout, QHBoxLayout,
    QSizePolicy, QListWidget, QListWidgetItem,
    QFrame, QComboBox, QGraphicsOpacityEffect
)

ColorA = '#2E6082' # Darkest
ColorB = '#92CBEA' # Mid darkness
ColorC = '#C7E5F5' # Brightest


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VWCP")
        self.setGeometry(0, 0, 1600, 900)
        self.setWindowIcon(QIcon('deliverable/imgs/VWCPLogo.png'))

        # Title Box
        MasterLayout = QVBoxLayout()
        MasterLayout.addWidget(Header("Van der Waals EOS Predictor",
                                       25, 'white', ColorA, 40, '10', '10'))

        MasterLayout.addLayout(HeaderWidgetLayout( MoleculeInput(), "Molecule Input"))
        MasterLayout.addLayout(HeaderWidgetLayout( Output(), "Output"))

        widget = QWidget()
        widget.setLayout(MasterLayout)
        widget.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(ColorC))
        widget.setPalette(palette)
        self.setCentralWidget(widget)

class Header(QLabel): #Creates all of the section Headers as well as main label
    def __init__(self, text, textSize=25, color='red', backgroundColor='blue', height=50, topRadius='0', bottomRadius='0'):
        super().__init__()
        self.setText(text)
        self.setFont(QFont("Baskerville Old Face", textSize))
        self.setStyleSheet("color: " + color + ";"
                           "background-color: " + backgroundColor + ";"
                           "font-weight: bold;"
                           "border-top-left-radius: " + topRadius + "px;"
                           "border-top-right-radius: " + topRadius + "px;"
                           "border-bottom-left-radius: " + bottomRadius + "px;"
                           "border-bottom-right-radius: " + bottomRadius + "px;")
        self.setAlignment(Qt.AlignCenter)
        self.setMaximumHeight(height)

class HeaderWidgetLayout(QVBoxLayout): # Returns a layout with appropriate header and widget
    def __init__(self, widget, text):
        super().__init__()
        self.setSpacing(0)
        self.addWidget(Header(text, 20, 'white', ColorA, 30, '10', '0'))
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.addWidget(widget)

class MoleculeInput(QWidget): # SMILES string input, generates molecular image
    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(ColorB))
        self.setPalette(palette)
        

        # SMILES string input box
        self.text_input = QLineEdit()
        self.text_input.setFont(QFont("Baskerville Old Face", 15))
        self.text_input.setFixedHeight(40)
        self.text_input.setFixedWidth(600)
        self.text_input.setAlignment(Qt.AlignCenter)

        #Generate image button
        self.button = QPushButton("Generate", self)
        # self.button.setFixedSize(150, 40)
        self.button.setFont(QFont("Baskerville Old Face", 20))
        self.button.setStyleSheet("color: " + ColorA + ";"
                             "background-color: " + ColorC + ";"
                             "font-weight: bold;")
        self.button.clicked.connect(self.generate_button)
        self.button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.text_input.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        text_input_layout = QHBoxLayout()
        text_input_layout.addWidget(self.text_input)
        text_input_layout.addWidget(self.button)
        text_input_layout.setAlignment(Qt.AlignHCenter)

        # Image display label
        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addLayout(text_input_layout)
        layout.addWidget(self.label)
        layout.setAlignment(self.label, Qt.AlignCenter)
        self.setLayout(layout)

    def generate_button(self): # Function runs when generate button is pressed
        # Takes text input, passes into rdkit, generates image based on SMILES
        try:
            smi = self.text_input.text()
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            PILimage = Draw.MolToImage(mol, size=(256,256))
            # MolBlock = Chem.MolToMolBlock(mol) For potential encoding
            image = QImage(PILimage.tobytes("raw", "RGB"), PILimage.width, PILimage.height, QImage.Format_RGB888)

            #Saves image and SMILES to data/files
            if image.save("deliverable/data/MolImage.png"):
                print("Image saved successfully")
            else: 
                print("Image failed to save")

            with open("deliverable/data/SMILES.txt", "w") as file:
                file.write(smi)
            
            #Applies the generated image to display label
            qpixmap = QPixmap.fromImage(image)
            qpixmap = qpixmap.scaled(300,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(qpixmap)
        except Exception as e:
            print(f"Invalid Smile String {e}")


# class Settings(QWidget): # Model selection for debugging purposes
#     def __init__(self, outputReference):
#         super().__init__()

#         self.setAutoFillBackground(True)
#         palette = QPalette()
#         palette.setColor(QPalette.Window, QColor(ColorB))
#         self.setPalette(palette)

#         self.encoder_list = {
#             'CNN1': {
#                     'desc': '3 Conv layers, featuring batchNorm and ReLU, to an fc pipeline. Output vector 64.',
#                     'path': 'deliverable/models/cnn_1.pth',
#                     'img_path': 'deliverable/imgs/cnn_architecture.png'
#             },
#             'CNN2': {
#                     'desc': 'Model description, whatever whatever blah blah',
#                     'path': 'deliverable/models/cnn_1.pth',
#                     'img_path': 'deliverable/imgs/cnn_architecture.png'
#             },
#             'CNN3': {
#                     'desc': 'Model description, whatever whatever blah blah',
#                     'path': 'deliverable/models/cnn_1.pth',
#                     'img_path': 'deliverable/imgs/cnn_architecture.png'
#             },
#         }
#         self.decoder_list = {
#             'None': {
#                     'desc': 'No decoder',
#                     'path': '',
#                     'img_path': 'deliverable/imgs/pkan.png'
#             },
#             'PKAN1': {
#                     'desc': 'Model description, whatever whatever blah blah',
#                     'path': 'deliverable/models/pkan_model.pth',
#                     'img_path': 'deliverable/imgs/pkan.png'
#             },
#             'PKAN2': {
#                     'desc': 'Model description, whatever whatever blah blah',
#                     'path': 'deliverable/models/pkan_model.pth',
#                     'img_path': 'deliverable/imgs/pkan.png'
#             },
#             'PKAN3': {
#                     'desc': 'Model description, whatever whatever blah blah',
#                     'path': 'deliverable/models/pkan_model.pth',
#                     'img_path': 'deliverable/imgs/pkan.png'
#             },
#         }

#         self.encoder_select = QComboBox()
#         self.encoder_select.addItems(self.encoder_list.keys())
#         self.decoder_select = QComboBox()
#         self.decoder_select.addItems(self.decoder_list.keys())


#         self.encoder_display = Display_Model(True, self.encoder_select, self.encoder_list, outputReference)
#         self.decoder_display = Display_Model(False, self.decoder_select, self.decoder_list, outputReference)

#         encoder_label = QLabel("Encoder:")
#         encoder_label.setStyleSheet(f"font-weight: bold; font-size: 18px; color: {ColorA}; padding: 5px;")
#         encoder_label.setAlignment(Qt.AlignLeft)
#         encoder_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

#         decoder_label = QLabel("Decoder:")
#         decoder_label.setStyleSheet(f"font-weight: bold; font-size: 18px; color: {ColorA}; padding: 5px;")
#         decoder_label.setAlignment(Qt.AlignLeft)
#         decoder_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

#         layout = QVBoxLayout()
#         layout.addWidget(encoder_label)
#         layout.addWidget(self.encoder_display)
#         layout.addWidget(decoder_label)
#         layout.addWidget(self.decoder_display)

#         self.setLayout(layout)
        

# class Display_Model(QWidget):
#     def __init__(self, isEncoder, comboBox, ModelList, outputReference):
#         super().__init__()

#         self.setAutoFillBackground(True)
#         palette = QPalette()
#         palette.setColor(QPalette.Window, QColor(ColorC))
#         self.setPalette(palette)

#         self.isEncoder = isEncoder
#         self.comboBox = comboBox
#         self.modelList = ModelList
#         self.outputReference = outputReference
#         comboBox.currentIndexChanged.connect(self.updateDisplay)

#         self.img   = QLabel()
#         self.desc  = QLabel()
#         self.desc.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
#         self.desc.setWordWrap(True)

#         self.updateDisplay()

#         layout = QHBoxLayout()
#         layout.addWidget(self.img)
#         layout.addWidget(self.desc)
#         layout.addWidget(self.comboBox)
#         self.setLayout(layout)

#     def updateDisplay(self):
#         selectedModel = self.comboBox.currentText()
#         desc = self.modelList[selectedModel]['desc']
#         path = self.modelList[selectedModel]['path']
#         self.outputReference.update_path(self.isEncoder, path)
#         img_path = self.modelList[selectedModel]['img_path']

#         self.img.setFixedSize(160,120)
#         self.img.setPixmap(QPixmap(img_path).scaled(160, 120, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
#         self.desc.setText(desc)

class Output(QWidget): # Takes the output from MoleculeInput and passes to model scripts
    def __init__(self):
        super().__init__()

        self.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(ColorB))
        self.setPalette(palette)

        self.encoder_path = ""
        self.decoder_path = ""

        # Run button
        self.button = QPushButton("Run", self)
        self.button.setFixedSize(150, 40)
        self.button.setFont(QFont("Baskerville Old Face", 20))
        self.button.setStyleSheet("color: " + ColorA + ";"
                             "background-color: " + ColorC + ";"
                             "font-weight: bold;")
        self.button.clicked.connect(self.run_button)

        # Output text box, displays stdout of ran file
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        self.output_text.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Preferred
        )
        self.output_text.setStyleSheet('background-color: ' + ColorC + ';')
        self.output_text.setFontPointSize(15)

        # Logo over output box
        img_label = QLabel()
        img_label.setPixmap(QPixmap('deliverable/imgs/VWCPLogo.png').scaled(300,300,Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(.25)
        img_label.setGraphicsEffect(opacity_effect)
        box_layout = QHBoxLayout()
        box_layout.addWidget(img_label)
        box_layout.setAlignment(img_label, Qt.AlignHCenter)
        self.output_text.setLayout(box_layout)

        self.left_graph = GraphDisplay(0)
        self.right_graph = GraphDisplay(1)

        # Graph Image Output
        graph_layout = QHBoxLayout()
        graph_layout.addWidget(self.left_graph)
        graph_layout.addWidget(self.right_graph)

        display_layout = QHBoxLayout()
        display_layout.addLayout(graph_layout)
        display_layout.addWidget(self.output_text)

        layout = QVBoxLayout()
        layout.addLayout(display_layout)
        layout.addWidget(self.button)
        layout.setAlignment(self.button, Qt.AlignHCenter)
        self.setLayout(layout)

    def update_path(self, isEncoder, path):
        if (isEncoder): 
            self.encoder_path = path
            print(path)
        else:
            self.decoder_path = path
            print(path)

    
    def run_button(self):
        #Runs specified file stdout is sent to handle_output
        self.output_text.clear()
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.finished.connect(self.script_finished)
        self.process.start("python", ["-u", "deliverable/predict.py", 'deliverable/data/MolImage.png', 'deliverable/data/SMILES.txt'])

    def handle_output(self): # Takes ran file stdout and appends it to display box
        output = self.process.readAllStandardOutput().data().decode('utf-8')
        self.output_text.append(output)

    def script_finished(self): # Placeholder in case I want after completion code to run
        self.left_graph.update_graph()
        self.right_graph.update_graph()

class GraphDisplay(QWidget):
    def __init__(self, default):
        super().__init__()
        self.label = QLabel()
        self.select = QComboBox()
        self.graph_list = {
            'Predictor A': 'deliverable/data/Predictor_a.png',
            'Predictor B': 'deliverable/data/Predictor_b.png',
            'RandomForestModel A': 'deliverable/data/RandomForestModel_a.png',
            'RandomForestModel B': 'deliverable/data/RandomForestModel_b.png',
            'LinearRegressor A': 'deliverable/data/LinearRegressor_a.png',
            'LinearRegressor B': 'deliverable/data/LinearRegressor_b.png',
            'MLP A': 'deliverable/data/MLP_a.png',
            'MLP B': 'deliverable/data/MLP_b.png',
            'PINN A': 'deliverable/data/PINN_a.png',
            'PINN B': 'deliverable/data/PINN_b.png',
            'PKAN A': 'deliverable/data/PKAN_a.png',
            'PKAN B': 'deliverable/data/PKAN_b.png',
            'CNN A': 'deliverable/data/CNN_a.png',
            'CNN B': 'deliverable/data/CNN_b.png',
        }
        self.select.addItems(self.graph_list.keys())
        self.select.setCurrentIndex(default)
        self.select.currentIndexChanged.connect(self.update_graph)

        self.img_ref = 'deliverable/data/MolImage.png'
        layout = QVBoxLayout()
        layout.addWidget(self.select)
        layout.addWidget(self.label)
        layout.setAlignment(self.label, Qt.AlignCenter)
        self.setLayout(layout)
    def update_graph(self):
        self.img_ref = self.graph_list[self.select.currentText()]
        self.label.setPixmap(QPixmap(self.img_ref).scaled(300,300,Qt.IgnoreAspectRatio, Qt.SmoothTransformation))



class Color(QWidget): # Test widget for layout debugging
    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)
        size_policy = self.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()