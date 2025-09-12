;uses raw-y-data and maps it to an average velocity of 100, everything above can reach up to 127, anything below down to 50

(defun bird-vel (ys)
  (let* ((avg (/ (float (reduce #'+ ys)) (length ys)))
         (max-diff (apply #'max (mapcar (lambda (y) (abs (- y avg))) ys))))
    (mapcar (lambda (y)
              (let* ((diff (abs (- y avg)))
                     (scale (/ diff (max 1.0 max-diff)))
                     (vel (+ 100 (* scale (if (> y avg) -50 27)))))
                (max 50 (min 127 (round vel)))))
            ys)))


