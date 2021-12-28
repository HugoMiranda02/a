function clear() {
  $("#btnHome").parent().removeClass("active");
  $("#btnProd").parent().removeClass("active");
  $("#btnEstatisticas").parent().removeClass("active");
  $("#btnManutencao").parent().removeClass("active");
  $("#btnEngenharia").parent().removeClass("active");
  $("#btnCalibracao").parent().removeClass("active");
  $("#btnFabricante").parent().removeClass("active");
  $("#btnEngenhariaProdutos").parent().removeClass("active");
  $("#btnConfig").parent().removeClass("active");
  $("#btnTestePerifericos").parent().removeClass("active");
  $("#btnAccess").parent().removeClass("active");
  $("#mainHeader").slideUp();
  $("#mainContent").css("min-height", "88vh");
  $("a[aria-expanded=true]").attr("aria-expanded", "false");
  $(".collapse").removeClass("show");
}

function toggleDropDown() {}

$("#btnHome").on("click", function () {
  clear();
  $("#mainHeader").slideDown();
  $("#mainContent").css("min-height", "74vh");
  $("#pageName").text("Home");
  changePage("/home");
  $(this).parent().addClass("active");
});

$("#btnProd").on("click", function () {
  clear();
  $("#pageName").text("Produção");
  changePage("/ferramentas");
  $(this).parent().addClass("active");
});
