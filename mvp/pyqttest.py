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
    QFrame
)
#import py3Dmol

ColorA = '#2E6082' # Darkest
ColorB = '#92CBEA' # Mid darkness
ColorC = '#C7E5F5' # Brightest


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VWCP")
        self.setGeometry(0, 0, 1600, 900)
        self.setWindowIcon(QIcon('mvp/VWCPLogo.png'))

        # Title Box
        MasterLayout = QVBoxLayout()
        MasterLayout.addWidget(Header("Van der Waals EOS Predictor",
                                       25, 'white', ColorA, 40, '10', '10'))

        # Molecule Input
        LowerLayout = QHBoxLayout()
        LowerLayout.addLayout(HeaderWidgetLayout( MoleculeInput(), "Molecule Input"))

        # Right Side
        SideLayout = QVBoxLayout()
        self.settings = Settings()
        self.output = Output(self.settings.encoder_selector, self.settings.decoder_selector)
        SideLayout.addLayout(HeaderWidgetLayout( self.settings, "Control"))
        SideLayout.addLayout(HeaderWidgetLayout( self.output, "Output"))

        MasterLayout.addLayout(LowerLayout)
        LowerLayout.addLayout(SideLayout)

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
            PILimage = Draw.MolToImage(mol, size=(300,300))
            # MolBlock = Chem.MolToMolBlock(mol) For potential encoding
            image = QImage(PILimage.tobytes("raw", "RGB"), PILimage.width, PILimage.height, QImage.Format_RGB888)

            #Saves image and SMILES to data/files
            if image.save("mvp/data/MolImage.png"):
                print("Image saved successfully")
            else: 
                print("Image failed to save")

            with open("mvp/data/SMILES.txt", "w") as file:
                file.write(smi)
            
            #Applies the generated image to display label
            qpixmap = QPixmap.fromImage(image)
            qpixmap = qpixmap.scaled(500,500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(qpixmap)
        except Exception as e:
            print(f"Invalid Smile String {e}")


class Settings(QWidget): # Model selection for debugging purposes
    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(ColorB))
        self.setPalette(palette)

        self.encoder_selector = Model_Selector(True)
        self.decoder_selector = Model_Selector(False)

        layout = QVBoxLayout()
        layout.addLayout(HeaderWidgetLayout(self.encoder_selector, 'Encoder Selector'))
        layout.addLayout(HeaderWidgetLayout(self.decoder_selector, 'Decoder Selector'))

        self.setLayout(layout)

class Model_Selector(QWidget):
    def __init__(self, is_encoder):
        super().__init__()

        self.encoder_list = {
            'CNN1': {
                    'desc': '3 Conv layers, featuring batchNorm and ReLU, to an fc pipeline. Output vector 64.',
                    'path': 'mvp/models/cnn_1.pth',
                    'img_path': 'mvp/imgs/cnn_architecture.png'
            },
            'CNN2': {
                    'desc': 'Model description, whatever whatever blah blah',
                    'path': 'mvp/models/cnn_1.pth',
                    'img_path': 'mvp/imgs/cnn_architecture.png'
            },
            'CNN3': {
                    'desc': 'Model description, whatever whatever blah blah',
                    'path': 'mvp/models/cnn_1.pth',
                    'img_path': 'mvp/imgs/cnn_architecture.png'
            },
        }
        self.decoder_list = {
            'PKAN1': {
                    'desc': 'Model description, whatever whatever blah blah',
                    'path': 'mvp/models/pkan_model.pth',
                    'img_path': 'mvp/imgs/pkan.png'
            },
            'PKAN2': {
                    'desc': 'Model description, whatever whatever blah blah',
                    'path': 'mvp/models/pkan_model.pth',
                    'img_path': 'mvp/imgs/pkan.png'
            },
            'PKAN3': {
                    'desc': 'Model description, whatever whatever blah blah',
                    'path': 'mvp/models/pkan_model.pth',
                    'img_path': 'mvp/imgs/pkan.png'
            },
        }

        self.model_list = self.encoder_list if is_encoder else self.decoder_list
        self.selected_item = None
        self.selected_model_path = None

        self.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(ColorC))
        self.setPalette(palette)

        self.model_list_widget = QListWidget()
        self.model_list_widget.setStyleSheet(f'background-color: {ColorA};')

        layout = QVBoxLayout()
        layout.addWidget(self.model_list_widget)
        self.setLayout(layout)

        self.list_models()

    def list_models(self):

        for model_name, model_info in self.model_list.items():

            item = QListWidgetItem(self.model_list_widget)
            model_item = Model_Item(model_name, model_info['desc'], model_info['path'], model_info['img_path'], self)

            item.setSizeHint(model_item.sizeHint())
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)
            self.model_list_widget.addItem(item)
            self.model_list_widget.setItemWidget(item, model_item)

    def update_selection(self, selected_item, model_path, checked):
        if checked:
            if self.selected_item and self.selected_item is not selected_item:
                self.selected_item.check_button.setChecked(False)
            self.selected_item = selected_item
            self.selected_model_path = model_path
        else:
            if self.selected_item is selected_item:
                self.selected_item = None
                self.selected_model_path = None

    def get_selected_model_path(self):
        return self.selected_model_path

class Model_Item(QWidget):
    def __init__(self, name, desc, model_path, img_path, parent):
        self.parent = parent
        self.model_path = model_path
        self.img_path = img_path
        super().__init__()
        self.setStyleSheet(f"background-color: {ColorC};")

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)

        # === Header (Model Name) ===
        header = QLabel(name)
        header.setStyleSheet(f"font-weight: bold; font-size: 18px; color: white; padding: 5px;")
        header.setAlignment(Qt.AlignLeft)

        # === Divider Line ===
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        # === Image and Description Section ===
        content_layout = QHBoxLayout()

        # Placeholder Image
        self.image_label = QLabel("img")
        self.image_label.setFixedSize(160, 120)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap(img_path).scaled(160, 120, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

        # Description
        self.desc_label = QLabel(f"{desc}")
        self.desc_label.setStyleSheet("font-size: 12px; color: white; padding: 5px;")

        button_widget = QWidget()
        button_layout = QVBoxLayout()
        button_widget.setStyleSheet(f"background-color:{ColorC};")
        
        self.check_button = QPushButton('')
        self.check_button.setCheckable(True)
        self.check_button.setStyleSheet(f"background-color:{ColorA};")
        self.check_button.clicked.connect(self.toggle_selection)
        
        button_layout.addWidget(self.check_button)
        button_widget.setLayout(button_layout)

        content_layout.addWidget(self.image_label, 1)
        content_layout.addWidget(self.desc_label, 3)
        content_layout.addWidget(button_widget)

        # === Add Everything to Main Layout ===
        main_layout.addWidget(header, 1)
        main_layout.addWidget(divider, 1)
        main_layout.addLayout(content_layout, 3)

        self.setLayout(main_layout)

    def toggle_selection(self, checked):
        self.parent.update_selection(self, self.model_path, checked)

class Output(QWidget): # Takes the output from MoleculeInput and passes to model scripts
    def __init__(self, encoder_selector, decoder_selector):
        super().__init__()

        self.encoder_selector = encoder_selector
        self.decoder_selector = decoder_selector

        self.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(ColorB))
        self.setPalette(palette)

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

        layout = QVBoxLayout()
        layout.addWidget(self.output_text)
        layout.addWidget(self.button)
        layout.setAlignment(self.button, Qt.AlignHCenter)
        self.setLayout(layout)
    
    def run_button(self):
        #Runs specified file stdout is sent to handle_output
        self.output_text.clear()

        encoder_path = self.encoder_selector.get_selected_model_path()
        decoder_path = self.decoder_selector.get_selected_model_path()

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.finished.connect(self.script_finished)
        self.process.start("python", ["-u", "mvp/predict.py", 'mvp/data/MolImage.png', encoder_path, decoder_path])

    def handle_output(self): # Takes ran file stdout and appends it to display box
        output = self.process.readAllStandardOutput().data().decode('utf-8')
        self.output_text.append(output)

    def script_finished(self): # Placeholder in case I want after completion code to run
        pass


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