#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Rod {
private:
    double E;
    double rho;
    double S;
    double L;
    double pLimit;
    double dLimit;

    vector<double> x;
    vector<double> v;
    double m;
    double h;

    double max_global_eps;
    double max_local_eps;
    bool is_damaged;

public:
    Rod(double E, double rho, double S, double L, double pLimit, double dLimit)
        : E(E), rho(rho), S(S), L(L), pLimit(pLimit), dLimit(dLimit) {
    }

    void mesh(unsigned int n) {
        x.resize(n + 1);
        v.resize(n + 1);

        m = S * L * rho / n;
        h = L / n;

        for(unsigned int i = 0; i <= n; ++i) {
            x[i] = -L + i * h;
        }

        max_global_eps = 0;
        max_local_eps = 0;
        is_damaged = false;
    }

    void strike(double V) {

        for(unsigned int i = 0; i < x.size(); i++) {
            v[i] = V;
        }

        double c = sqrt(E/rho);
        double tau = 0.95 * h / c;
        double maxT = 2 * L / c;
        double T = 0;

        while (T < maxT) {
            for(unsigned int i = 0; i < x.size(); i++) {
                double eps_l = i > 0 ? (x[i] - x[i-1] - h) / h : 0;
                double eps_r = i < x.size() - 1 ? (x[i+1] - x[i] - h) / h : 0;

                double local_eps = max(abs(eps_l), abs(eps_r));
                max_local_eps = max(max_local_eps, local_eps);
                if(local_eps * E > dLimit) {
                    is_damaged = true;
                }

                double F = (-eps_l + eps_r) * E * S;

                double a = F / m;
                v[i] += a * tau;
                if((i == x.size() - 1) && v[i] > 0 ) {
                    v[i] = 0;
                }
            }

            for(unsigned int i = 0; i < x.size(); i++) {
                x[i] += v[i] * tau;
            }

            T += tau;
            max_global_eps = max(max_global_eps, abs(x[x.size() - 1] - x[0] - L) / L);

            //cout << "T = " << T << " L = " << (x[x.size() - 1] - x[0]) << " eps = " << max_global_eps  << endl;
        }
    }

    bool damaged() const {
        return is_damaged;
    }

    double max_eps() const {
        return max_global_eps;
    }
};


int main() {
    Rod r(2e11 /*0.5e11 4e11*/, 7800 /*1500 9000*/, 1e-4, 1.0, 2.5e8, 4e8 /*0.5e8 4e8*/);
    r.mesh(100);
    r.strike(5); // 1-10
    cout << boolalpha << "Is damaged: " << r.damaged() << " Max eps: " << r.max_eps() << endl;
    return 0;
}