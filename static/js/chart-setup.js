function setupComparisonChart(labels, realData, predData) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Réel',
            data: realData,
            borderColor: '#0d6efd',
            tension: 0.3,
            pointRadius: 2,
            fill: false
          },
          {
            label: 'Prédit',
            data: predData,
            borderColor: '#fd7e14',
            tension: 0.3,
            pointRadius: 2,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' }
        },
        scales: {
          x: {
            title: { display: true, text: 'Index' }
          },
          y: {
            title: { display: true, text: 'Global Active Power (kW)' }
          }
        }
      }
    });
  }