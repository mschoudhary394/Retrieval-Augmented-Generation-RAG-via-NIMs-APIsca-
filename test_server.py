from flask import Flask, request, jsonify
import itertools
from utility import get_response
#from utility2 import get_response
app = Flask(__name__) 

memory = 0

#curl -XPOST -H "Content-Type: application/json" -d '{"question": "What is Cuda?"}' http://dgx04:6002/complete
@app.route('/complete', methods=['POST'])
def complete_endpoint():
    global memory
    data = request.get_json()
    question = data.get('question')
    if question:
        memory=1
        answer, context =get_response(question)
        return jsonify({"answer":answer, "contexts":[context]})
    return None
    
    
    """if question:
        words = question.split() # Split input string into words
        answer = ' '.join(sorted(words, key=len)) # Sort words by length and join them back together
        result=get_response(answer)
        contexts = [' '.join(p) for p in itertools.permutations(question.split())]
        memory=1
        return jsonify({'answer': result, 'contexts': contexts})
    return jsonify({'error': 'Invalid input'}), 400"""

# curl -XGET -H "Content-Type: application/json" -d '{"reset": 1}' http://dgx04:6002/reset
@app.route('/reset', methods=['GET'])
def reset_endpoint():
    global memory
    data=request.get_json()
    flag = data.get('reset')
    if flag==1:
        memory = 0
        return jsonify({'reset_response': 'success'})
    else:
        return jsonify({'reset_response': 'failure'})
    return jsonify({'error': 'Invalid inpuxt'}), 400
 
if __name__ == '__main__':
    app.run(debug=True,port=6002,host='0.0.0.0')
