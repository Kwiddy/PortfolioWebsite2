const app = require('./app');
console.log("Server starting on port 8998...")

app.listen(process.env.PORT || 8998);