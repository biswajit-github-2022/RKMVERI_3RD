<?php
require_once('connect.php');

// Project 1: Search for stars
$input_text = '';
$search_result = null;

if (isset($_POST['search_submit'])) {
    $input_text = $_POST['starname'];
    $search_query = "SELECT title, starname, role FROM starsin WHERE title LIKE '%$input_text%';";
    $search_result = mysqli_query($con, $search_query);

    if (!$search_result) {
        die("Query failed: " . mysqli_error($con));
    }
}

// Project 2: Fetch movie titles
$movies_query = "SELECT title FROM movies";
$movies_result = mysqli_query($con, $movies_query);

if (!$movies_result) {
    die("Query failed: " . mysqli_error($con));
}

// Add review form submission
// if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['review_submit'])) {
//     $title = mysqli_real_escape_string($con, $_POST['title']);
//     $user = mysqli_real_escape_string($con, $_POST['user']);
//     $review = mysqli_real_escape_string($con, $_POST['review']);
    
//     $review_query = "INSERT INTO Reviews (title, user, review) VALUES ('$title', '$user', '$review')";
    
//     if (mysqli_query($con, $review_query)) {
//         echo "<script>alert('Review added successfully!'); window.location.href='".$_SERVER['PHP_SELF']."';</script>";

//     } else {
//         echo "<script>alert('Error: " . mysqli_error($con) . "'); window.location.href='index.php';</script>";
//     }
// }

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['review_submit'])) {
    $title = mysqli_real_escape_string($con, $_POST['title']);
    $user = mysqli_real_escape_string($con, $_POST['user']);
    $review = mysqli_real_escape_string($con, $_POST['review']);
    
    // Check if a record with the same title and user already exists
    $check_query = "SELECT * FROM Reviews WHERE title = '$title' AND user = '$user'";
    $check_result = mysqli_query($con, $check_query);

    if (mysqli_num_rows($check_result) > 0) {
        // Update the review field if the row exists
        $update_query = "UPDATE Reviews SET review = '$review' WHERE title = '$title' AND user = '$user'";
        if (mysqli_query($con, $update_query)) {
            echo "<script>alert('Your data is updated!'); window.location.href='" . $_SERVER['PHP_SELF'] . "';</script>";
        } else {
            echo "<script>alert('Error: " . mysqli_error($con) . "'); window.location.href='index.php';</script>";
        }
    } else {
        // Insert a new record if no matching row exists
        $review_query = "INSERT INTO Reviews (title, user, review) VALUES ('$title', '$user', '$review')";
        if (mysqli_query($con, $review_query)) {
            echo "<script>alert('Review added successfully!'); window.location.href='" . $_SERVER['PHP_SELF'] . "';</script>";
        } else {
            echo "<script>alert('Error: " . mysqli_error($con) . "'); window.location.href='index.php';</script>";
        }
    }
}

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Portal</title>
    <link rel="stylesheet" href="style.css">
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(to bottom right, #2099e9, #fa40cb);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #333;
        }

        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            width: 90%;
            max-width: 1200px;
            background: #fff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }

        .section {
            width: 48%;
            padding: 10px;
        }

        h2 {
            color: #2099e9;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        table,
        th,
        td {
            border: 1px solid #ddd;
        }

        th,
        td {
            padding: 8px;
            text-align: left;
        }

        th {
            background: #2099e9;
            color: #fff;
        }

        input,
        select,
        textarea,
        button {
            width: 100%;
            padding: 8px;
            margin: 8px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        button {
            background: #fa40cb;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #e836b3;
        }

        .new {
            display: flex;
            flex-direction: column;
        }
        .head{
            color: white;
            font-size:70px;
            font-weight:bold;
        }

    </style>
</head>
<body>
    <div>
        <h1 class="head" >
            Movie DataBase
        </h1>
    </div>
    <div class="container">
        <!-- Section for Search -->
        <div class="section">
            <h2>Search for Stars</h2>
            <form method="post" action="">
                <label for="starname">Enter Starname Substring:</label>
                <input type="text" id="starname" name="starname" value="<?php echo htmlspecialchars($input_text); ?>" required>
                <button type="submit" name="search_submit">Submit</button>
            </form>
            <?php if ($search_result && mysqli_num_rows($search_result) > 0): ?>
            <h3>Matching Stars</h3>
            <table>
                <tr>
                    <th>Title</th>
                    <th>Starname</th>
                    <th>Role</th>
                </tr>
                <?php while ($row = mysqli_fetch_assoc($search_result)): ?>
                <tr>
                    <td><?php echo htmlspecialchars($row['title']); ?></td>
                    <td><?php echo htmlspecialchars($row['starname']); ?></td>
                    <td><?php echo htmlspecialchars($row['role']); ?></td>
                </tr>
                <?php endwhile; ?>
            </table>
            <?php elseif (isset($_POST['search_submit'])): ?>
            <p>No results found for "<?php echo htmlspecialchars($input_text); ?>"</p>
            <?php endif; ?>
        </div>

        <!-- Section for Add Review -->
        <div class="section">
            <h2>Add a Review</h2>
            <form method="post" action="" class="new">
                <label for="movie-title" class="form-label">Select Movie:</label>
                <select id="movie-title" name="title" required>
                    <option value="" disabled selected>Select a Movie</option>
                    <?php while ($row = mysqli_fetch_assoc($movies_result)) { ?>
                        <option value="<?php echo htmlspecialchars($row['title']); ?>">
                            <?php echo htmlspecialchars($row['title']); ?>
                        </option>
                    <?php } ?>
                </select>
                
                <label for="user-id">Your User ID:</label>
                <input type="text" id="user-id" name="user" placeholder="Enter your User ID" required>
                
                <label for="review-text">Your Review:</label>
                <textarea id="review-text" name="review" rows="8" placeholder="Write your review here..." required></textarea>
                
                <button type="submit" name="review_submit">Submit Review</button>
            </form>
        </div>
    </div>
</body>
</html>
