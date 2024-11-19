// Fetch the weights from the Python server
fetch('http://127.0.0.1:5000/receive', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        variable: [
            { 1: [1, 2, 3, 4, 5, 6, 7, 8, 9] },
            { 2: [9, 8, 7, 6, 5, 4, 3, 2, 1] },
            { 3: [2, 4, 6, 8, 10, 12, 14, 16, 18] },
            { 4: [3, 6, 9, 12, 15, 18, 21, 24, 27] },
            { 9: [4, 8, 12, 16, 20, 24, 28, 32, 36] }
        ]
    }) // Replace this `variable` with actual input data
})
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            console.log("Optimal weights:", data.weights);

            // Optionally, store the weights globally or manipulate them
            const weights = data.weights;

            // Example: Save weights to localStorage (optional)
            localStorage.setItem('weights', JSON.stringify(weights));
        } else {
            console.error("Error:", data.error);
        }
    })
    .catch(error => {
        console.error("Error:", error);
    });
