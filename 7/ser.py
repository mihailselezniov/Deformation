from flask import Flask, request, jsonify, make_response
from fiber_test2 import Test_drive_fiber


app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/test_drive_fiber', methods=['POST'])
def test():
    return jsonify({'results': Test_drive_fiber(request.json).get_results()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
#sudo pip3 install flask
