		plt.bar(x, y1, 0.6, color='r')
        plt.bar(x, y2_m, 0.6, bottom=y1, color='b')
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig("window_and_elec_count.png")

        # Gaussian distribution of elec and window count
        plot = Gaussian.plot(np.mean(y1), np.std(y1), "elec_count")
        plot = Gaussian.plot(np.mean(y2), np.std(y2), "window_count")
        fig2 = plt.gcf()
        plt.show()
        fig2.savefig("Gaussian_window_and_elec_count.png")

        # Plot histogram of window and elec count
        plt.bar(y2, y1, width=2, align='center')  # A bar chart
        fig3 = plt.gcf()
        plt.xlabel('window_count')
        plt.ylabel('elec_count')
        plt.show()
        fig3.savefig("Histogram_window_and_elec_count.png")
