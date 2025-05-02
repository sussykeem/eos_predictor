import sys
import os
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QPalette, QImage
from PyQt5.QtCore import Qt, QProcess
from rdkit import Chem
from rdkit.Chem import Draw
from PyQt5.QtWidgets import ( 
    QApplication, QMainWindow, 
    QLabel, QWidget, QPushButton, QTextEdit, QLineEdit,
    QVBoxLayout, QHBoxLayout,
    QSizePolicy, QComboBox, QGraphicsOpacityEffect
)

ColorA = '#2E6082' # Darkest
ColorB = '#92CBEA' # Mid darkness
ColorC = '#C7E5F5' # Brightest

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(sys.prefix, "Library", "plugins", "platforms")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VWCP")
        self.setGeometry(0, 0, 1600, 900)
        self.setWindowIcon(QIcon('imgs/VWCPLogo.png'))

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
            if image.save("data/MolImage.png"):
                print("Image saved successfully")
            else: 
                print("Image failed to save")

            with open("data/SMILES.txt", "w") as file:
                file.write(smi)
            
            #Applies the generated image to display label
            qpixmap = QPixmap.fromImage(image)
            qpixmap = qpixmap.scaled(300,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(qpixmap)
        except Exception as e:
            print(f"Invalid Smile String {e}")

class Output(QWidget): # Takes the output from MoleculeInput and passes to model scripts
    def __init__(self):
        super().__init__()

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

        # Logo over output box
        img_label = QLabel()
        img_label.setPixmap(QPixmap('imgs/VWCPLogo.png').scaled(300,300,Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
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

    
    def run_button(self):
        #Runs specified file stdout is sent to handle_output
        self.output_text.clear()
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.finished.connect(self.script_finished)
        self.process.start("python", ["-u", "predict.py", 'data/MolImage.png', 'data/SMILES.txt'])

    def handle_output(self): # Takes ran file stdout and appends it to display box
        output = self.process.readAllStandardOutput().data().decode('utf-8')
        self.output_text.append(output)

    def script_finished(self): # Placeholder in case I want after completion code to run
        final_display = self.output_text.toPlainText()

        #self.output_text.clear()
        self.left_graph.update_graph()
        self.right_graph.update_graph()

class GraphDisplay(QWidget): # Displays Graphs generated by the predictor
    def __init__(self, default):
        super().__init__()
        self.label = QLabel()

        # Drop-down selection menu
        self.select = QComboBox()
        self.graph_list = {
            'Predictor A': 'data/Predictor_a.png',
            'Predictor B': 'data/Predictor_b.png',
            'RandomForestModel A': 'data/RandomForestModel_a.png',
            'RandomForestModel B': 'data/RandomForestModel_b.png',
            'LinearRegressor A': 'data/LinearRegressor_a.png',
            'LinearRegressor B': 'data/LinearRegressor_b.png',
            'MLP A': 'data/MLP_a.png',
            'MLP B': 'data/MLP_b.png',
            'PINN A': 'data/PINN_a.png',
            'PINN B': 'data/PINN_b.png',
            'PKAN A': 'data/PKAN_a.png',
            'PKAN B': 'data/PKAN_b.png',
            'CNN A': 'data/CNN_a.png',
            'CNN B': 'data/CNN_b.png',
        }
        self.select.addItems(self.graph_list.keys())
        self.select.setCurrentIndex(default)
        self.select.currentIndexChanged.connect(self.update_graph)

        self.img_ref = 'data/MolImage.png'
        layout = QVBoxLayout()
        layout.addWidget(self.select)
        layout.addWidget(self.label)
        layout.setAlignment(self.label, Qt.AlignCenter)
        self.setLayout(layout)
    def update_graph(self): # Refreshes displayed graph, changes on run button or selection change
        self.img_ref = self.graph_list[self.select.currentText()]
        self.label.setPixmap(QPixmap(self.img_ref).scaled(400,400,Qt.IgnoreAspectRatio, Qt.SmoothTransformation))



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