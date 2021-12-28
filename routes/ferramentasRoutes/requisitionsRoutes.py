from flask import Blueprint, Response, request, jsonify
from SistemaVisao import vision, MainFilters, MainProperties
import os
import sqlite3
requisitionsRoutes = Blueprint("requisitionsRoutes", __name__)

ds_factor = 0.6


gray = False
roi = False

cam = vision()


def gen(cam):
    while True:
        frame = cam.preview()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_img(cam):
    while True:
        frame = cam.view()
        if frame != "":
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@requisitionsRoutes.route('/video_feed')
def video_feed():
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@requisitionsRoutes.route('/img_feed')
def img_feed():
    return Response(gen_img(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@requisitionsRoutes.route('/ROI', methods=["POST"])
def roi():
    data = request.form
    x = data.get("x")
    y = data.get("y")
    w = data.get("w")
    h = data.get("h")
    MainFilters["roi"] = (x, y, w, h)
    return ''


@requisitionsRoutes.route('/blur', methods=["POST"])
def blur():
    data = request.form
    blur = data.get("blur")
    MainFilters["blur"] = blur
    return ''


@requisitionsRoutes.route('/getEdges', methods=["GET"])
def getEdges():
    return jsonify(MainFilters["edges"])


@requisitionsRoutes.route('/kernelX', methods=["POST"])
def kernelX():
    data = request.form
    kernel = data.get("kernel")
    MainFilters["edges"]["kernelx"] = kernel
    return ''


@requisitionsRoutes.route('/kernelY', methods=["POST"])
def kernelY():
    data = request.form
    kernel = data.get("kernel")
    MainFilters["edges"]["kernely"] = kernel
    return ''


@requisitionsRoutes.route('/kernelFX', methods=["POST"])
def kernelFX():
    data = request.form
    kernel = data.get("kernel")
    MainFilters["edges"]["kernelFx"] = kernel
    return ''


@requisitionsRoutes.route('/kernelFY', methods=["POST"])
def kernelFY():
    data = request.form
    kernel = data.get("kernel")
    MainFilters["edges"]["kernelFy"] = kernel
    return ''


@requisitionsRoutes.route('/thresh', methods=["POST"])
def thresh():
    data = request.form
    thresh = data.get("thresh")
    MainFilters["edges"]["thresh1"] = thresh
    return ''


@requisitionsRoutes.route('/threshUpper', methods=["POST"])
def threshupper():
    data = request.form
    thresh = data.get("thresh")
    MainFilters["edges"]["thresh2"] = thresh
    return ''


@requisitionsRoutes.route('/edges', methods=["POST"])
def edges():
    data = request.form
    edges = data.get("edges")
    MainFilters["edges"]["enable"] = edges == "true"
    return ''


@requisitionsRoutes.route('/internal', methods=["POST"])
def internal():
    data = request.form
    edges = data.get("edges")
    MainFilters["edges"]["internal"] = edges == "true"
    MainFilters["edges"]["external"] = False
    return ''


@requisitionsRoutes.route('/external', methods=["POST"])
def external():
    data = request.form
    edges = data.get("edges")
    MainFilters["edges"]["external"] = edges == "true"
    MainFilters["edges"]["internal"] = False
    return ''


@requisitionsRoutes.route('/selectPixel', methods=["POST"])
def pixel():
    data = request.form
    x = data.get("x")
    y = data.get("y")
    MainFilters["pixels"] = (int(x), int(y),)
    return ''


@requisitionsRoutes.route('/reset', methods=["GET"])
def reset():
    MainFilters["roi"] = False
    MainFilters["blur"] = False
    MainFilters["pixels"] = False
    MainFilters["edges"]["enable"] = False
    MainFilters["edges"]["kernelx"] = 3
    MainFilters["edges"]["kernely"] = 3
    MainFilters["edges"]["kernelFx"] = 3
    MainFilters["edges"]["kernelFy"] = 3
    MainFilters["edges"]["thresh1"] = 127
    MainFilters["edges"]["thresh2"] = 255
    MainFilters["edges"]["external"] = True
    MainFilters["edges"]["Internal"] = False
    return ''


@requisitionsRoutes.route('/trigger', methods=["GET"])
def trigger():
    cam.trigger()
    return ''


def getProdExist(codigo):
    find_prod = f"SELECT id FROM ferramentas WHERE nome = '{codigo}'"
    prod_exists = 0
    with sqlite3.connect("db/main.db") as con:
        try:
            cursor = con.cursor()
            products = cursor.execute(find_prod)
            for _ in products:
                prod_exists = 1
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Algo deu errado, tente novamente.'})
    cursor.close()
    con.close()
    return (prod_exists)


@ requisitionsRoutes.route('/verify', methods=["POST"])
def verify():
    data = request.form
    productExist = getProdExist(data.get('nome'))
    if productExist == 1:
        return jsonify({"result": 0, "msg": "Já existe uma ferramenta com este nome!"})

    # Checa se foi inserido código de produto
    if data.get('nome') == '':
        return jsonify({"result": 0, "msg": "Insira o nome da ferramenta!"})

    # Checa se foi inserido descrição
    if data.get('desc') == '':
        return jsonify({"result": 0, "msg": "Insira a descrição da ferramenta!"})

    return jsonify({"result": 1})


@ requisitionsRoutes.route('/getFerramentas', methods=["GET"])
def getProductlist():
    get_products = 'SELECT * from ferramentas'
    with sqlite3.connect("db/main.db") as con:
        try:
            cursor = con.cursor()
            products = cursor.execute(get_products)
            products_list = {}
            for product in products:
                products_list[len(products_list)] = product
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Algo deu errado, tente novamente.'}), 500
    cursor.close()
    con.close()
    return jsonify(products_list), 200

# ********  Retorna produto requisitado  ********


@ requisitionsRoutes.route("/getFerramenta/<id>", methods=["GET"])
def getProduct(id):
    with sqlite3.connect("db/main.db") as con:
        get_product = f"select filtro, valor from filtros where ferramenta = {id}"
        try:
            cursor = con.cursor()
            query = cursor.execute(get_product)
            produtos = query.fetchall()
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Produto não existe!'}), 404
    cursor.close()
    con.close()
    print(produtos)
    left = 0
    top = 0
    w = 0
    h = 0
    for i in produtos:
        if i[0] == "blur":
            MainFilters["blur"] = i[1]
        if i[0] == "left":
            left = i[1]
        if i[0] == "top":
            top = i[1]
        if i[0] == "w":
            w = i[1]
        if i[0] == "h":
            h = i[1]

    if left > -1 and top > -1 and w > -1 and h > -1:
        MainFilters["roi"] = (left, top, w, h)
    return jsonify(produtos), 200

# ********  Cadastra produto  ********


@ requisitionsRoutes.route("/registrarFerramenta", methods=["POST"])
def cadastraProduct():

    # Checa se foi produto já foi cadastrado
    data = request.form
    productExist = getProdExist(data.get('nome'))
    if productExist == 1:
        return jsonify({"result": 0, "msg": "Já existe uma ferramenta com este nome!"})

    # Checa se foi inserido código de produto
    if data.get('nome') == '':
        return jsonify({"result": 0, "msg": "Insira o nome da ferramenta!"})

    # Checa se foi inserido descrição
    if data.get('desc') == '':
        return jsonify({"result": 0, "msg": "Insira a descrição da ferramenta!"})
    cadastra_prod = f''' INSERT INTO ferramentas VALUES (
                    NULL,
                    '{data.get('nome')}' ,
                    '{data.get('desc')}'
                )'''

    with sqlite3.connect("db/main.db") as con:
        try:
            cursor = con.cursor()
            cursor.execute(cadastra_prod)
            con.commit()
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Algo deu errado, tente novamente.'})

    ultimoId = cursor.lastrowid
    MainProperties["ferramenta"]["id"] = ultimoId
    con.close()
    return jsonify({"result": 1, "id": ultimoId})


@ requisitionsRoutes.route("/salvarFiltros", methods=["POST"])
def salvaFerramenta():
    data = request.form.getlist("filtros[]")
    id = request.form.get('id')
    data = data[0].split(",")
    delete = f"DELETE FROM filtros WHERE ferramenta = {id}"
    with sqlite3.connect("db/main.db") as con:
        try:
            cursor = con.cursor()
            cursor.execute(delete)
            con.commit()
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Algo deu errado, tente novamente.'})
    for _ in range(len(data)//2):
        cadastra_prod = f''' INSERT INTO filtros VALUES (
                        NULL,
                        '{data[0]}',
                        {data[1]},
                        {id}
                    )'''

        with sqlite3.connect("db/main.db") as con:
            try:
                cursor = con.cursor()
                cursor.execute(cadastra_prod)
                con.commit()
            except Exception as e:
                print(e)
                return jsonify({'msg': 'Algo deu errado, tente novamente.'})
        data.pop(0)
        data.pop(0)
        con.close()

    MainProperties["ferramenta"]["save_preview"] = True
    return jsonify({"result": 1, "id": "ultimoId"})


@ requisitionsRoutes.route("/deletaFerramenta/<id>", methods=["POST"])
def deleteFerramenta(id):

    deletaFerramenta = f''' DELETE FROM ferramentas WHERE id = {id}'''

    delete = f"DELETE FROM filtros WHERE ferramenta = {id}"
    with sqlite3.connect("db/main.db") as con:
        try:
            cursor = con.cursor()
            cursor.execute(delete)
            con.commit()
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Algo deu errado, tente novamente.'})

    with sqlite3.connect("db/main.db") as con:
        try:
            cursor = con.cursor()
            cursor.execute(deletaFerramenta)
            con.commit()
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Algo deu errado, tente novamente.'})

    os.remove(f"static/imgs/{id}")

    ultimoId = cursor.lastrowid
    con.close()
    return jsonify({"result": 1, "id": ultimoId})


@ requisitionsRoutes.route("/updateFerramenta/<id>", methods=["POST"])
def updateFerramenta(id):
    data = request.form

    if data.get('nome') == '':
        return jsonify({"result": 0, "msg": "Insira o nome da ferramenta!"})

    # Checa se foi inserido descrição
    if data.get('desc') == '':
        return jsonify({"result": 0, "msg": "Insira a descrição da ferramenta!"})

    updateFerramenta = f''' UPDATE ferramentas SET nome = "{data.get("nome")}", desc = "{data.get("desc")} WHERE id = {id}"'''

    with sqlite3.connect("db/main.db") as con:
        try:
            cursor = con.cursor()
            cursor.execute(updateFerramenta)
            con.commit()
        except Exception as e:
            print(e)
            return jsonify({'msg': 'Algo deu errado, tente novamente.'})

    ultimoId = cursor.lastrowid
    con.close()
    return jsonify({"result": 1, "msg": "Sucesso!"})
