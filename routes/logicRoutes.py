from flask import Blueprint, jsonify
from SistemaVisao import MainProperties, MainFilters
logicRoutes = Blueprint("logicRoutes", __name__)


@logicRoutes.route("/mainUpdate", methods=["GET"])
def x():
    return jsonify({"properties": MainProperties, "filters": MainFilters}), 200
