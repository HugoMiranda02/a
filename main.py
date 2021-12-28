from flask import Flask
from routes.pagesRoutes import pageRoutes
from routes.logicRoutes import logicRoutes
from routes.ferramentasRoutes.requisitionsRoutes import requisitionsRoutes

app = Flask(__name__)
app.register_blueprint(logicRoutes)
app.register_blueprint(pageRoutes)
app.register_blueprint(requisitionsRoutes)


UPLOAD_FOLDER = 'static/imgs/products'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(debug=False, host='0.0.0.0')
