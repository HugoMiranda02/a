from flask import Blueprint, jsonify
from SistemaVisao import MainProperties
logicRoutes = Blueprint("logicRoutes", __name__)


@logicRoutes.route("/mainUpdate", methods=["GET"])
def x():
    return jsonify(MainProperties), 200
