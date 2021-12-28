from flask import Blueprint
from SistemaVisao import MainProperties
from flask.templating import render_template
pageRoutes = Blueprint("pageRoutes", __name__)


@pageRoutes.route("/", methods=['GET'])
def aplication():
    return render_template("index.html")


@pageRoutes.route("/ferramentas", methods=['GET'])
def ferramentas():
    return render_template("pages/registroFerramenta.html")


@pageRoutes.route("/ferramenta", methods=['GET'])
def visao():
    return render_template("pages/visaoComputacional.html")


@pageRoutes.route("/production/<id>", methods=['GET'])
def producao(id):
    MainProperties["ferramenta"]["id"] = int(id)
    return render_template("pages/producao.html")


@pageRoutes.route("/filtro/<page>", methods=['GET'])
def filtros(page):
    return render_template(f"filtros/{page}.html")
