<!-- <?php
echo "Hello, World!";
echo "hi";
?> -->



 <?php
// Database configuration
$servername = "localhost";
$username = "biswajit"; // Replace with your MySQL username
$password = "biswajit_mysql"; // Replace with your MySQL password
$dbname = "grocery_store"; // Replace with your database name

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// SQL query to select data
$sql = "SELECT * FROM items"; // Replace with your table name
$result = $conn->query($sql);

// HTML for displaying data
echo "<!DOCTYPE html>";
echo "<html>";
echo "<head>";
echo "<title>MySQL Data Display</title>";
echo "<style>";
echo "table { width: 100%; border-collapse: collapse; }";
echo "th, td { border: 1px solid black; padding: 8px; text-align: left; }";
echo "th { background-color: #f2f2f2; }";
echo "</style>";
echo "</head>";
echo "<body>";
echo "<h1>Data from MySQL Table</h1>";

if ($result->num_rows > 0) {
    echo "<table>";
    echo "<tr>";
    // Output table headers
    $fields = $result->fetch_fields();
    foreach ($fields as $field) {
        echo "<th>" . htmlspecialchars($field->name) . "</th>";
    }
    echo "</tr>";

    // Output data rows
    while ($row = $result->fetch_assoc()) {
        echo "<tr>";
        foreach ($row as $column) {
            echo "<td>" . htmlspecialchars($column) . "</td>";
        }
        echo "</tr>";
    }
    echo "</table>";
} else {
    echo "0 results";
}

// Close connection
$conn->close();

echo "</body>";
echo "</html>";
?>
