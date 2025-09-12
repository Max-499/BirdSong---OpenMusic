(defun add-or-subtract (a b)
  (if (zerop (random 2))           
      (+ a b)                 
      (- a b)))                   


(defun seq-append (seqs)
  "Nimmt eine Liste von chord-seqs und hängt sie zeitlich hintereinander."
  (let ((all-chords '())
        (onset 0))
    (dolist (cs seqs)
      ;; Hole die Chords mit ihren lokalen Onsets
      (let* ((chords (om::get-chords cs))
             (ons    (om::get-onsets cs))
             (durs   (om::get-durations cs)))
        (dotimes (i (length chords))
          (push (list (nth i chords)
                      (+ onset (nth i ons))
                      (nth i durs))
                all-chords))
        ;; Nach dem Seq kommt der neue Onset ans Ende
        (setf onset (+ onset (reduce #'+ durs)))))
    ;; Aus den gesammelten Daten wieder chord-seq bauen
    (let ((sorted (sort all-chords #'< :key #'second)))
      (om::make-chord-seq
       :chords    (mapcar #'first sorted)
       :onsets    (mapcar #'second sorted)
       :durations (mapcar #'third sorted)))))
