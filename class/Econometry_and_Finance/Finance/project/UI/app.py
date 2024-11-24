# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/receive', methods=['POST'])
# def receive_variable():
#     data = request.json
#     print("Received variable:", data['variable'])
#     return jsonify({"status": "success", "received": data['variable']})

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from cvxopt import matrix, solvers

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/receive', methods=['POST'])
def receive_variable():
    # Step 1: Receive JSON data
    data = request.json.get('variable')
    if not data or len(data) != 5:
        return jsonify({"error": "Invalid data received. Expected 5 rows."}), 400

    # Step 2: Create a 9x5 matrix
    data_matrix = np.zeros((9, 5))  # Initialize 9x5 matrix with zeros
    for col, col_data in enumerate(data):
        key = list(col_data.keys())[0]  # Extract the column key
        values = col_data[key]         # Extract the array values
        if len(values) == 9:           # Ensure the array length is 9
            data_matrix[:, col] = values
        else:
            return jsonify({"error": f"Column {key} does not have 9 values."}), 400

    # Step 3: Perform Variance-Covariance and Optimization
    try:
        # Variance-Covariance Matrix
        cov_matrix = np.cov(data_matrix, rowvar=False)

        # Mean Vector
        mean_vector = np.mean(data_matrix, axis=0)

        # Define constraints for Quadratic Programming
        num_assets = cov_matrix.shape[0]
        Dmat = 2 * cov_matrix  # Quadratic term
        dvec = np.zeros(num_assets)  # Linear term
        A = np.ones((1, num_assets))  # Equality constraint (sum(weights) = 1)
        b = np.array([1.0])
        G = -np.eye(num_assets)  # Inequality constraint (weights >= 0)
        h = np.zeros(num_assets)

        # Convert to cvxopt matrices
        Dmat = matrix(Dmat)
        dvec = matrix(dvec)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        # Solve Quadratic Programming Problem
        solution = solvers.qp(Dmat, dvec, G, h, A, b)
        weights = np.array(solution['x']).flatten()

        # Return response
        return jsonify({
            "status": "success",
            "weights": weights.tolist(),  # Convert to list for JSON serialization
            "mean_vec": mean_vector.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
