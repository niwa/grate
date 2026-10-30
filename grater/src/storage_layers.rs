use ndarray::{s, Zip, Array1, Array2, Array4, ArrayView2};


/// Holds sprop across all chainages in (nc, nlayers, nbins, nlith) array
///
/// It would be handy if `[:, 0, :, :]` was the top storage layer for transport
/// and output. However because we are adding and removing layers there would
/// be a lot of copying to maintain that. Rather, keep an array of indices to
/// the top storage layer and treat the 4d array as a circular list in the
/// nlayers dimension. Ie suppose we have 3 layers called a, b, and c at a
/// chainage (doesn't matter which one):
///     [, a, :, :]
///     [, b, :, :]
///     [, c, :, :]
/// and head = [...0,...] indicating layer a is at the top at this chainage.
/// We add a layer, call it d, so b and c must be combined:
///     [, a, :, :]
///     [, b+c, :, :]
///     [, c, :, :]
/// and now the third row can be reused as the top layer
///     [, a, :, :]
///     [, b+c, :, :]
///     [, d, :, :]
/// and head is updated to [...2,...]. We read this dimension circularly thus
/// have layers d, a, b+c.
///
/// Similar process when deleting a layer, except we must update `count` so we
/// know that layer isn't in use. Eg suppose we continue with the
/// above, but delete layer d (since it is the top one), we have layers a and
/// b+c left
///     [, a, :, :]
///     [, b+c, :, :]
///     [, ?, :, :]
/// with head back to [...0,...] and count [...2,...]
pub struct StorageLayers {
    nc: usize,
    max_nlayers: usize,
    max_thick: f64,
    sprop: Array4<f64>,             // (nc, max_nlayers, nbins, nlith)
    thickness: Array2<f64>,         // (nc, max_nlayers)
    count: Array1<usize>,           // number of storage layers at each chainage, (nc,)
    head: Array1<usize>,            // index to top storage layer, (nc,)
}

impl StorageLayers {

    pub fn new(max_thick: f64, sprop: Array4<f64>, thickness: Array2<f64>) -> Self {
        let (nc, nlayers) = thickness.dim();
        let count: Array1::<usize> = thickness.rows().into_iter().map(|row| row.iter().filter(|&&x| x > 0.0).count()).collect();
        let head = Array1::<usize>::zeros(nc);

        Self {
            nc,
            max_nlayers: nlayers,
            max_thick,
            sprop,
            thickness,
            count,
            head,
        }
    }

    /// Top layer sprop at chainage c, shape (nbins, nlith)
    pub fn get_top_sprop_at(&self, c: usize) -> ArrayView2<'_, f64> {
        self.sprop.slice(s![c, self.head[c], .., ..])
    }

    /// Return thicknesses in logical top-to-bottom order, padding inactive layers with NaN.
    pub fn get_thickness(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::from_elem(self.thickness.dim(), f64::NAN);
        for c in 0..self.nc {
            for k in 0..self.count[c] {
                let idx = (self.head[c] + k) % self.max_nlayers;
                out[[c, k]] = self.thickness[[c, idx]];
            }
        }
        out
    }

    /// Return sprop in logical top-to-bottom order, padding inactive layers with NaN.
    pub fn get_sprop(&self) -> Array4<f64> {
        let mut out = Array4::<f64>::from_elem(self.sprop.dim(), f64::NAN);
        for c in 0..self.nc {
            for k in 0..self.count[c] {
                let idx = (self.head[c] + k) % self.max_nlayers;
                out.slice_mut(s![c, k, .., ..])
                    .assign(&self.sprop.slice(s![c, idx, .., ..]));
            }
        }
        out
    }

    /// Insert a new uninitialied top layer at chainage c.
    fn add_top_layer(&mut self, c: usize, new_thickness: f64) {
        let n = self.max_nlayers;
        assert!(n > 1, "Cannot add a layer if max number of storage layers is {n}");
        let old_head = self.head[c];
        let new_head = (old_head + n - 1) % n;

        if self.count[c] == n {
            // mix the bottom two layers to free a slot
            let bottom = (old_head + self.count[c] - 1) % n;
            let above_bottom = (old_head + self.count[c] - 2) % n;
            let tup = self.thickness[[c, above_bottom]];
            let tbot = self.thickness[[c, bottom]];

            let bottom_s = self.sprop.slice(s![c, bottom, .., ..]).to_owned();

            if tbot.is_infinite() {
                self.thickness[[c, above_bottom]] = f64::INFINITY;
                self.sprop
                    .slice_mut(s![c, above_bottom, .., ..])
                    .assign(&bottom_s);
            } else {
                let total = tup + tbot;
                assert!(total > 0.0);
                self.sprop
                    .slice_mut(s![c, above_bottom, .., ..])
                    .zip_mut_with(&bottom_s, |a, &b| {
                        *a = (*a * tup + b * tbot) / total;
                    });
                self.thickness[[c, above_bottom]] = total;
            }

            self.count[c] -= 1;
        }

        // slot immediately before the old head is available
        self.head[c] = new_head;

        // new layer has undefined composition, so just leave it
        self.thickness[[c, new_head]] = new_thickness;
        self.count[c] += 1;
    }

    /// Aggrade top layer, proportions and thickness at a particular chainage
    ///
    /// c: chain idx
    /// aprop: view into active layer proportions (nbins, nlith)
    /// my_jli: view into active layer transport rate (nbins, nlith)
    /// my_sum: sum of my_jli over both dims
    /// chi: chi in interface function
    /// dy: increase by this amount
    ///
    pub fn aggrade_at(&mut self, c: usize, aprop: &ArrayView2<f64>, my_jli: &ArrayView2<f64>, my_sum: f64, chi: f64, dy: f64) {
        let mut change = dy;

        if change <= 0.0 {
            return;
        }

        if change.abs() > self.max_thick {
            eprintln!(
                "warning: dy[{}]={} exceeds the allowed range (-{}, {}); capping it.",
                c, change, self.max_thick, self.max_thick
            );
            change = change.clamp(-self.max_thick, self.max_thick);
        }

        let i = self.head[c];
        let thick = self.thickness[[c, i]];
        let fill = if self.max_nlayers == 1 {
            // must fill the only layer with change
            change
        } else {
            // if thick is inf, then fill is -inf, so we add a layer
            change.min(self.max_thick - thick)
        };

        // fill top layer
        if fill > 0.0 {
            // only change proportions if finite thickness
            if ! thick.is_infinite() {
                Zip::from(&mut self.sprop.slice_mut(s![c, i, .., ..]))
                   .and(aprop)
                   .and(my_jli)
                   .for_each(|sv, &a, &my| {
                       let f_int = if my_sum == 0.0 {
                           a
                       } else {
                           chi * a + (1.0 - chi) * (my / my_sum)
                       };
   
                       *sv = (*sv * thick + f_int * fill) / (thick + fill);
                   });
            }
            self.thickness[[c, i]] = thick + fill;
            change -= fill;
        }

        // remaining aggradation forms new top layer
        if change > 0.0 {
            self.add_top_layer(c, change);
            let i = self.head[c];
            Zip::from(&mut self.sprop.slice_mut(s![c, i, .., ..]))
                .and(aprop)
                .and(my_jli)
                .for_each(|sv, &a, &my| {
                    *sv = if my_sum == 0.0 {
                        a
                    } else {
                        chi * a + (1.0 - chi) * (my / my_sum)
                    };
                });
        }
    }

    /// Degrade top layer by dy
    pub fn degrade_at(&mut self, c: usize, dy: f64) {
        let mut change = dy;

        if change >= 0.0 {
            return;
        }

        if change.abs() > self.max_thick {
            eprintln!(
                "warning: dy[{}]={} exceeds the allowed range (-{}, {}); capping it.",
                c, change, self.max_thick, self.max_thick
            );
            change = change.clamp(-self.max_thick, self.max_thick);
        }

        // degradation: remove material from the top
        let mut remaining = -change;

        while remaining > 0.0 && self.count[c] > 0 {
            let i = self.head[c];
            let thick = self.thickness[[c, i]];

            // can't degrade, we are done
            if thick.is_infinite() {
                break;
            }

            let removed = remaining.min(thick);
            let new_thick = thick - removed;
            self.thickness[[c, i]] = new_thick;
            remaining -= removed;

            // more to go?
            if new_thick <= 0.0 {
                self.thickness[[c, i]] = 0.0;
                self.head[c] = (i + 1) % self.max_nlayers;
                self.count[c] -= 1;
            }
        }
    }

}

