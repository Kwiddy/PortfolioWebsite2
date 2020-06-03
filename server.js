const app = require('./app').app;
console.log("Server starting on port 8888...")

app.listen(process.env.PORT || 8888);