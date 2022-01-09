const loginForm = document.getElementById("login-form");
const loginButton = document.getElementById("login-form-submit");
const loginErrorMsg = document.getElementById("login-error-msg");


loginButton.addEventListener("click", (e) => {
    e.preventDefault();
    const username = loginForm.username.value;
    const password = loginForm.password.value;

    let data = {"name": username,
      "pass": password};

    fetch("http://10.0.0.111:5000/new_user", {
        method: "POST",
        mode: "no-cors",
        headers: {'Content-Type': 'application/json'}, 
        body: JSON.stringify(data)
      }).then(res => {
        console.log("Request complete! response:", res);
      });

    window.location.replace("http://127.0.0.1:5500/web-app/frontend/main_page.html");

});


