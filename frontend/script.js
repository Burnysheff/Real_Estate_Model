document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  const resultDiv = document.getElementById('result');

  form.addEventListener('submit', async event => {
    event.preventDefault();

    const formData = new FormData(form);
    const data = {};
    formData.forEach((val, key) => {
      if (key === 'has_balcony' || key === 'has_parking') {
        data[key] = form[key].checked;
      }
      else {
        data[key] = isNaN(val) ? val : +val;
      }
    });

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const json = await response.json();

      if (!response.ok) {
        throw new Error(json.detail || 'Unknown error');
      }

      const pred = json.price_rub;
      let html = `<p>Предсказанная цена: <strong>${Math.round(pred).toLocaleString()} ₽</strong></p>`;

      if (data.real_price) {
        const diff = pred - data.real_price;
        const sign = diff >= 0 ? '+' : '–';
        html += `<p>Отклонение от введённой: <strong>${sign}${Math.abs(Math.round(diff)).toLocaleString()} ₽</strong></p>`;
      }
      resultDiv.innerHTML = html;
    }
    catch (err) {
      resultDiv.innerHTML = `<p style="color:red">Ошибка: ${err.message}</p>`;
    }
  });
});
