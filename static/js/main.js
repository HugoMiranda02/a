var MainProperties = {};
var Ferramenta = {
  id: 0,
  nome: "",
  desc: "",
};

const myTimer = async () => {
  await $.ajax({
    url: "/mainUpdate",
    method: "GET",
    data: {},
    dataType: "JSON",
    success: function (data) {
      MainProperties = data;
    },
    error: function (data) {
      console.log(data);
    },
  });
};

setInterval(() => myTimer(), 100);

function changePage(url) {
  $("#mainContent").load(url, function () {});
  $("#sidebar").removeClass("active");
  $(".overlay").removeClass("active");
}

function changeFootText(text) {
  $("#footerText").text(text);
}
