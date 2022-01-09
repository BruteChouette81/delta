const Button = document.getElementById("test-button");

Button.addEventListener("click", (e) => {
    e.preventDefault();
    let data = {"name": "Thomas",
        "pass": "douce123"};

    fetch("http://10.0.0.111:5000/login", {
            method: "POST",
            mode: "no-cors",
            headers: {'Content-Type': 'application/json'}, 
            body: JSON.stringify(data)
        }).then(res => {
            if (res["log"] == true) {
                window.location.replace("http://127.0.0.1:5500/web-app/frontend/main_page.html")
            }
            else {
                alert("not good password or user...")
            }
        });

});