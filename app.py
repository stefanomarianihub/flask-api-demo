from flask import Flask, request, jsonify

app = Flask(__name__)

from sentence_transformers import SentenceTransformer, util
modelPath = "./model"

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#model.save(modelPath)
model = SentenceTransformer(modelPath)

@app.route('/test', methods = ["GET", "POST"])
def test():
    if request.method == "GET":
        return jsonify({"response": "Get Request Called"})
    elif request.method == "POST":
        req_json = request.json
        sent1 = req_json["sentence 1"]
        sent2 = req_json["sentence 2"]

        sentences1 = [sent1]

        sentences2 = [sent2]

        # Compute embedding for both lists
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        sim = "{:.4f}".format(cosine_scores[0][0])

        return jsonify({

            "sentence 1": sent1,
            "sentence 2": sent2,
            "similarity of the two sentences": sim

        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
