import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib as mpl
from matplotlib import colors

# Load data - vægt data (kvinder/mænd)
data = np.loadtxt('height_weight.csv', delimiter=';', skiprows=1)
X = data[:,1:3]
y = data[:,0]

menInd = np.where(y == 0)
womenInd = np.where(y == 1)

men = X[menInd,:]
women = X[womenInd,:]

# oevelse a
clf = QuadraticDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

mean = clf.means_
covariance = clf.covariance_
priors = clf.priors_

x_men = men[:,:,0].flatten()
y_men = men[:,:,1].flatten()
x_men, y_men = np.mgrid[np.min(x_men):np.max(x_men), np.min(y_men):np.max(y_men)]
pos_men = np.empty(x_men.shape + (2,))
pos_men[:, :, 0] = x_men; pos_men[:, :, 1] = y_men
rv_men = multivariate_normal(mean[0], covariance[0])
plt.figure(0)
plt.contourf(x_men, y_men, rv_men.pdf(pos_men))
plt.xlabel('højde i inches')
plt.ylabel('vægt i pund')

x_women = women[:,:,0].flatten()
y_women = women[:,:,1].flatten()
x_women, y_women = np.mgrid[np.min(x_women):np.max(x_women), np.min(y_women):np.max(y_women)]
pos_women = np.empty(x_women.shape + (2,))
pos_women[:, :, 0] = x_women; pos_women[:, :, 1] = y_women
rv_women = multivariate_normal(mean[1], covariance[1])
plt.figure(1)
plt.contourf(x_women, y_women, rv_women.pdf(pos_women))
plt.xlabel('højde i inches')
plt.ylabel('vægt i pund')

# oevelse b
menPdfCentroid = rv_men.pdf(mean[0])
# menPdfCentroid = np.max(rv_men.pdf(pos_men))

# Oevelse c
# Udfra figurerne for henholdsvis kvinder og mænd kan det ses at en højde på 190 cm (74,8")
# og en vægt på 30 kg (66 lbs) slet ikke vil være i nærheden af klokkekurverne. Derfor
# kan der defineres et threshhold for hvor lille en sandsynlighed der må være for en given værdi
# for at den skal defineres som en outlier

outlier = rv_men.pdf([74.8, 66]) # Næsten nul - vil kunne detekteres som en outlier

# Oevelse d

# #############################################################################
# Colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


# #############################################################################

# Plot functions
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(1, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis')
        plt.ylabel('Data with\n fixed covariance')
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis')
    elif fig_index == 3:
        plt.ylabel('Data with\n varying covariances')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')

    return splot


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='black', linewidth=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.2)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariance_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariance_[1], 'blue')


plt.figure(figsize=(10, 8), facecolor='white')
plt.suptitle('Linear Discriminant Analysis vs Quadratic Discriminant Analysis',
             y=0.98, fontsize=15)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
splot = plot_data(lda, X, y, y_pred, fig_index=1)
plot_lda_cov(lda, splot)
plt.axis('tight')

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
y_pred = qda.fit(X, y).predict(X)
splot = plot_data(qda, X, y, y_pred, fig_index=2)
plot_qda_cov(qda, splot)
plt.axis('tight')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

lda_error = 1 - lda.score(X,y)
qda_error = 1 - qda.score(X,y)

# Oevelse e
# 170 cm = 66.9", 80 kg = 176.4 lbs
woman_proba = rv_women.pdf([66.9, 176.4])
man_proba = rv_men.pdf([66.9, 176.4])