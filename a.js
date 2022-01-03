function addFilter(filter, value, filtros) {
  for (i in filtros) {
    if (filtros[i][0] == filter) {
      filtros[i] = [filter, value];
      return filtros;
    }
  }
  filtros = filtros.concat([filter, value]);
  return filtros;
}

var filtros = [["asd", 123]];

console.log(addFilter("asd", 123, filtros));
