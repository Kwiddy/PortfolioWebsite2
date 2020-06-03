const app = require('./app').app;
console.log("Server starting on port 8998...")

app.listen(process.env.PORT || 8998);