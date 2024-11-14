<?php
$servername = "localhost";  // Database server
$username = "root";         // Database username
$password = "root";             // Database password
$dbname = "your_database";  // Replace with your database name

// Create a connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Retrieve form data
$id = $_POST['id'];
$name = $_POST['name'];
$product_bought = $_POST['product_bought'];
$product_price = $_POST['product_price'];

// Prepare and execute the SQL insert query
$sql = "INSERT INTO users (id, name, product_bought, product_price) 
        VALUES ('$id', '$name', '$product_bought', '$product_price')";

if ($conn->query($sql) === TRUE) {
    echo "New record created successfully";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

// Close the connection
$conn->close();
?>
