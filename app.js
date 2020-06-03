const express = require('express');
const cors = require('cors');

const app = express();

app.use(express.static('Client')).use(cors());

module.exports = {app};